use crate::data::{Label, TextData};
use crate::text;
use crate::text::{possible_byte_substrings, possible_character_substrings};
use crate::unicode::{normalize, Normalization, CS};
use crate::utils::{accumulate, constrain, py_invalid_type_error, py_required_key_error};
use crate::whitespace::{find_substring_ignoring_whitespace, full, operations, remove};
use anyhow::anyhow;
use itertools::Itertools;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::ops::Sub;

pub type PreprocessingFn =
    dyn Send + Sync + 'static + Fn(TextData, Option<u64>) -> anyhow::Result<TextData>;
pub type LabelingFn = dyn Send + Sync + 'static + Fn(&TextData) -> anyhow::Result<Label>;

#[derive(PartialEq, Debug, Clone)]
pub enum PreprocessingConfig {
    // clean the sequences (remove spurious whitespaces)
    Clean(bool),
    // normalize the sequence
    // (using a unicode normalization scheme, see https://en.wikipedia.org/wiki/Unicode_equivalence#Normalization)
    Normalize(Normalization, bool),
    // overwrite original text with processed text
    Overwrite,
    // switch between multiple preprocessing functions
    Switch(Vec<PreprocessingConfig>, Vec<f64>),
    // delete all whitespaces
    NoWhitespaces(bool),
    // insert whitespaces between all characters
    FullWhitespaces(bool),
    // delete and insert whitespaces with certain probabilities
    NoiseWhitespaces(f64, f64, bool),
    // extract substrings from text
    CharSubstring(usize, bool),
    ByteSubstring(usize, bool),
    // randomly replace the language token with the given default
    LanguageDropout(f64),
}

impl IntoPy<PyObject> for PreprocessingConfig {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let d = PyDict::new(py);
        let preprocessing_type = match self {
            PreprocessingConfig::Clean(use_g) => {
                d.set_item("use_graphemes", use_g).unwrap();
                "clean"
            }
            PreprocessingConfig::Overwrite => "overwrite",
            PreprocessingConfig::Switch(configs, probs) => {
                let py_configs = PyList::empty(py);
                for config in configs {
                    py_configs.append(config.into_py(py)).unwrap();
                }
                d.set_item("configs", py_configs).unwrap();
                d.set_item("probabilities", PyList::new(py, probs)).unwrap();
                "switch"
            }
            PreprocessingConfig::NoWhitespaces(use_g) => {
                d.set_item("use_graphemes", use_g).unwrap();
                "no_whitespaces"
            }
            PreprocessingConfig::FullWhitespaces(use_g) => {
                d.set_item("use_graphemes", use_g).unwrap();
                "full_whitespaces"
            }
            PreprocessingConfig::NoiseWhitespaces(iw_p, dw_p, use_g) => {
                d.set_item("insert_whitespace_prob", iw_p).unwrap();
                d.set_item("delete_whitespace_prob", dw_p).unwrap();
                d.set_item("use_graphemes", use_g).unwrap();
                "noise_whitespaces"
            }
            PreprocessingConfig::CharSubstring(max_chars, use_g) => {
                d.set_item("max_chars", max_chars).unwrap();
                d.set_item("use_graphemes", use_g).unwrap();
                "char_substring"
            }
            PreprocessingConfig::ByteSubstring(max_bytes, use_g) => {
                d.set_item("max_bytes", max_bytes).unwrap();
                d.set_item("use_graphemes", use_g).unwrap();
                "byte_substring"
            }
            PreprocessingConfig::Normalize(scheme, use_g) => {
                d.set_item("scheme", scheme.into_py(py)).unwrap();
                d.set_item("use_graphemes", use_g).unwrap();
                "normalize"
            }
            PreprocessingConfig::LanguageDropout(p) => {
                d.set_item("prob", p).unwrap();
                "language_dropout"
            }
        };
        d.set_item("type", preprocessing_type).unwrap();
        d.to_object(py)
    }
}

impl<'a> FromPyObject<'a> for PreprocessingConfig {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let d: &PyDict = ob.extract()?;
        let Some(preprocessing_type) = d.get_item("type") else {
            return Err(py_required_key_error("type", "preprocessing config"));
        };
        let preprocessing_type: String = preprocessing_type.extract()?;
        let preprocessing_config = match preprocessing_type.as_str() {
            "clean" => {
                let use_graphemes = if let Some(value) = d.get_item("use_graphemes") {
                    value.extract()?
                } else {
                    true
                };

                PreprocessingConfig::Clean(use_graphemes)
            }
            "normalize" => {
                let use_graphemes = if let Some(value) = d.get_item("use_graphemes") {
                    value.extract()?
                } else {
                    true
                };

                let Some(scheme) = d.get_item("scheme") else {
                    return Err(py_required_key_error("scheme", "normalization config"));
                };
                PreprocessingConfig::Normalize(scheme.extract()?, use_graphemes)
            }
            "overwrite" => PreprocessingConfig::Overwrite,
            "no_whitespaces" => {
                let use_graphemes = if let Some(value) = d.get_item("use_graphemes") {
                    value.extract()?
                } else {
                    true
                };
                PreprocessingConfig::NoWhitespaces(use_graphemes)
            }
            "full_whitespaces" => {
                let use_graphemes = if let Some(value) = d.get_item("use_graphemes") {
                    value.extract()?
                } else {
                    true
                };
                PreprocessingConfig::FullWhitespaces(use_graphemes)
            }
            "noise_whitespaces" => {
                let use_graphemes = if let Some(value) = d.get_item("use_graphemes") {
                    value.extract()?
                } else {
                    true
                };

                let iw_p = if let Some(value) = d.get_item("insert_whitespace_prob") {
                    value.extract()?
                } else {
                    0.
                };
                let dw_p = if let Some(value) = d.get_item("delete_whitespace_prob") {
                    value.extract()?
                } else {
                    0.
                };
                PreprocessingConfig::NoiseWhitespaces(iw_p, dw_p, use_graphemes)
            }
            "switch" => {
                let Some(configs) = d.get_item("configs") else {
                    return Err(py_required_key_error("configs", "switch config"));
                };
                let Some(probs) = d.get_item("probabilities") else {
                    return Err(py_required_key_error("probabilities", "switch config"));
                };
                PreprocessingConfig::Switch(
                    configs
                        .extract::<&PyList>()?
                        .iter()
                        .map(|any| any.extract())
                        .collect::<PyResult<Vec<PreprocessingConfig>>>()?,
                    probs
                        .extract::<&PyList>()?
                        .iter()
                        .map(|any| any.extract())
                        .collect::<PyResult<Vec<f64>>>()?,
                )
            }
            "char_substring" => {
                let use_graphemes = if let Some(value) = d.get_item("use_graphemes") {
                    value.extract()?
                } else {
                    true
                };
                let Some(max_chars) = d.get_item("max_chars") else {
                    return Err(py_required_key_error("max_chars", "char substring config"));
                };
                PreprocessingConfig::CharSubstring(max_chars.extract()?, use_graphemes)
            }
            "byte_substring" => {
                let use_graphemes = if let Some(value) = d.get_item("use_graphemes") {
                    value.extract()?
                } else {
                    true
                };
                let Some(max_bytes) = d.get_item("max_bytes") else {
                    return Err(py_required_key_error("max_bytes", "byte substring config"));
                };
                PreprocessingConfig::ByteSubstring(max_bytes.extract()?, use_graphemes)
            }
            "language_dropout" => {
                let Some(p) = d.get_item("prob") else {
                    return Err(py_required_key_error("prob", "language dropout config"));
                };
                PreprocessingConfig::LanguageDropout(p.extract()?)
            }
            k => {
                return Err(py_invalid_type_error(k, "preprocessing"));
            }
        };
        Ok(preprocessing_config)
    }
}

#[derive(PartialEq, Debug, Clone)]
pub enum LabelingConfig {
    // generate whitespace correction labels given processed and original sequence
    LabelWhitespaceCorrection(bool),
}

impl IntoPy<PyObject> for LabelingConfig {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let d: &PyDict = PyDict::new(py);
        let labeling_type = match self {
            LabelingConfig::LabelWhitespaceCorrection(use_g) => {
                d.set_item("use_graphemes", use_g).unwrap();
                "whitespace_correction"
            }
        };
        d.set_item("type", labeling_type).unwrap();
        d.to_object(py)
    }
}

impl<'a> FromPyObject<'a> for LabelingConfig {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let d: &PyDict = ob.extract()?;
        let Some(labeling_type) = d.get_item("type") else {
            return Err(py_required_key_error("type", "labeling config"));
        };
        let labeling_type: String = labeling_type.extract()?;
        let labeling_config = match labeling_type.as_str() {
            "whitespace_correction" => {
                let use_graphemes = if let Some(value) = d.get_item("use_graphemes") {
                    value.extract()?
                } else {
                    true
                };
                LabelingConfig::LabelWhitespaceCorrection(use_graphemes)
            }
            k => {
                return Err(py_invalid_type_error(k, "labeling"));
            }
        };
        Ok(labeling_config)
    }
}

fn switch(fns: Vec<PreprocessingConfig>, probs: Vec<f64>) -> Box<PreprocessingFn> {
    let num_fns = fns.len();
    assert!(
        num_fns > 0 && num_fns == probs.len(),
        "expected one or more preprocessing for switch preprocessing and the same \
        number of probabilities"
    );
    // generate cumulative probabilities
    let cum_p: Vec<f64> = accumulate(&probs);
    // probabilities should sum to 1
    assert!(
        cum_p.last().copied().unwrap().sub(1f64).abs() < 1e-5,
        "all switch probabilities should sum to 1"
    );

    let fns: Vec<Box<PreprocessingFn>> = fns.into_iter().map(|f| preprocessing_fn(f)).collect();

    // return new function that switches between multiple preprocessing functions
    // based on the given probability distribution
    Box::new(move |item, seed| -> anyhow::Result<TextData> {
        let mut rng = if seed.is_some() {
            ChaCha8Rng::seed_from_u64(seed.unwrap())
        } else {
            ChaCha8Rng::from_entropy()
        };
        let r: f64 = rng.gen();
        let mut idx = 0;
        while idx < num_fns - 1 && r > cum_p[idx] {
            idx += 1;
        }
        fns[idx](item, seed)
    })
}

fn apply_to_text<F: Fn(&str) -> anyhow::Result<String> + Send + Sync + 'static>(
    f: F,
) -> Box<PreprocessingFn> {
    Box::new(move |item, _| {
        Ok(TextData {
            processed: f(&item.processed)?,
            ..item
        })
    })
}

fn overwrite_original_from_processed() -> Box<PreprocessingFn> {
    Box::new(|item, _| {
        Ok(TextData {
            original: item.processed.clone(),
            ..item
        })
    })
}

fn noise_whitespace(iw_p: f64, dw_p: f64, use_graphemes: bool) -> Box<PreprocessingFn> {
    let iw_p = constrain(iw_p, 0., 1.);
    let dw_p = constrain(dw_p, 0., 1.);
    assert!(
        iw_p > 0. || dw_p > 0.,
        "at least one of insert whitespace or delete whitespace probability must be greater 0"
    );
    Box::new(move |item, seed| {
        let mut rng = if seed.is_some() {
            ChaCha8Rng::seed_from_u64(seed.unwrap())
        } else {
            ChaCha8Rng::from_entropy()
        };
        let cs = CS::new(&item.processed, use_graphemes);
        let processed = cs
            .chars()
            .enumerate()
            .map(|(idx, c)| {
                let r: f64 = rng.gen();
                if c.is_whitespace() {
                    if r < dw_p {
                        "".to_string()
                    } else {
                        c.str.to_string()
                    }
                } else if r < iw_p && idx > 0 && !cs.get_char(idx - 1).is_whitespace() {
                    " ".to_string() + c.str
                } else {
                    c.str.to_string()
                }
            })
            .join("");
        Ok(TextData { processed, ..item })
    })
}

fn substring<F: Fn(&str) -> anyhow::Result<Vec<(usize, usize, usize)>> + Send + Sync + 'static>(
    name: String,
    substring_fn: F,
    use_graphemes: bool,
) -> Box<PreprocessingFn> {
    Box::new(move |item, seed| {
        let mut rng = if seed.is_some() {
            ChaCha8Rng::seed_from_u64(seed.unwrap())
        } else {
            ChaCha8Rng::from_entropy()
        };
        let possible_substrings = substring_fn(&item.processed)?;
        let idx = rng.gen_range(0..possible_substrings.len());
        let (start, end, _) = possible_substrings[idx];
        let processed = item.processed[start..end].to_string();
        let original =
            match find_substring_ignoring_whitespace(&item.original, &processed, use_graphemes) {
                Some(s) => s.trim().to_string(),
                None => {
                    return Err(anyhow!(
                        "original and processed sequences can only differ in \
            whitespaces when applying the {} substring preprocessing",
                        name
                    ))
                }
            };
        Ok(TextData {
            original,
            processed,
            ..item
        })
    })
}

fn char_substring(max_chars: usize, use_graphemes: bool) -> Box<PreprocessingFn> {
    substring(
        "character".to_string(),
        move |s| Ok(possible_character_substrings(s, max_chars, use_graphemes)),
        use_graphemes,
    )
}

fn byte_substring(max_bytes: usize, use_graphemes: bool) -> Box<PreprocessingFn> {
    substring(
        "byte".to_string(),
        move |s| Ok(possible_byte_substrings(s, max_bytes, use_graphemes)),
        use_graphemes,
    )
}

fn language_dropout(prob: f64) -> Box<PreprocessingFn> {
    let prob = constrain(prob, 0.0, 1.0);
    Box::new(move |item, seed| {
        let mut rng = if seed.is_some() {
            ChaCha8Rng::seed_from_u64(seed.unwrap())
        } else {
            ChaCha8Rng::from_entropy()
        };
        let r: f64 = rng.gen();
        if r < prob {
            Ok(TextData {
                language: None,
                ..item
            })
        } else {
            Ok(item)
        }
    })
}

fn preprocessing_fn(preprocessing: PreprocessingConfig) -> Box<PreprocessingFn> {
    match preprocessing {
        PreprocessingConfig::Clean(use_g) => apply_to_text(move |s| Ok(text::clean(s, use_g))),
        PreprocessingConfig::Overwrite => overwrite_original_from_processed(),
        PreprocessingConfig::Switch(fns, probs) => switch(fns, probs),
        PreprocessingConfig::NoiseWhitespaces(iw_p, dw_p, use_g) => {
            noise_whitespace(iw_p, dw_p, use_g)
        }
        PreprocessingConfig::NoWhitespaces(use_graphemes) => {
            apply_to_text(move |s| Ok(remove(s, use_graphemes)))
        }
        PreprocessingConfig::FullWhitespaces(use_graphemes) => {
            apply_to_text(move |s| Ok(full(s, use_graphemes)))
        }
        PreprocessingConfig::CharSubstring(l, use_graphemes) => char_substring(l, use_graphemes),
        PreprocessingConfig::ByteSubstring(l, use_graphemes) => byte_substring(l, use_graphemes),
        PreprocessingConfig::Normalize(scheme, use_graphemes) => {
            apply_to_text(move |s| Ok(normalize(s, scheme, use_graphemes)))
        }
        PreprocessingConfig::LanguageDropout(p) => language_dropout(p),
    }
}

pub fn preprocessing(preprocessing: Vec<PreprocessingConfig>) -> Box<PreprocessingFn> {
    // return new function that runs all given preprocessing functions
    // in order
    let fns: Vec<Box<PreprocessingFn>> = preprocessing
        .into_iter()
        .map(|p| preprocessing_fn(p))
        .collect();
    Box::new(move |mut item, seed| {
        for f in fns.iter() {
            item = f(item, seed)?;
        }
        Ok(item)
    })
}

fn whitespace_correction_label(use_graphemes: bool) -> Box<LabelingFn> {
    Box::new(move |item| {
        Ok(Label::SeqClassification(
            operations(&item.processed, &item.original, use_graphemes)?
                .into_iter()
                .map(|l| l as i32)
                .collect(),
        ))
    })
}

pub fn labeling(labeling: LabelingConfig) -> Box<LabelingFn> {
    match labeling {
        LabelingConfig::LabelWhitespaceCorrection(use_graphemes) => {
            whitespace_correction_label(use_graphemes)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::data::TextData;

    use super::noise_whitespace;

    #[test]
    fn test_noise_whitespace_preprocessing() -> anyhow::Result<()> {
        let noise_fn = noise_whitespace(0.0, 1.0, true);
        let data = TextData::new("a test".to_string(), None, None);
        let noised = noise_fn(data.clone(), Some(0))?;
        assert_eq!(&noised.processed, "atest");
        let noise_fn = noise_whitespace(1.0, 0.0, true);
        let data = TextData::new("a test".to_string(), None, None);
        let noised = noise_fn(data.clone(), Some(0))?;
        assert_eq!(&noised.processed, "a t e s t");
        let data = TextData::new("Ginsberǵs".to_string(), None, None);
        let noised = noise_fn(data.clone(), Some(0))?;
        assert_eq!(&noised.processed, "G i n s b e r ǵ s");
        Ok(())
    }
}
