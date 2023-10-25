use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::{
    tokenization::{tokenizer, TokenizerConfig},
    utils::{py_invalid_type_error, py_required_key_error},
    whitespace::operations,
};

use super::{Label, TextData};

pub type LabelingFn = dyn Send + Sync + 'static + Fn(&TextData) -> anyhow::Result<Label>;

#[derive(Debug, Clone)]
pub enum GenerationConfig {
    InputAndTarget(String),
    TargetOnly,
}

impl<'a> FromPyObject<'a> for GenerationConfig {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let d: &PyDict = ob.extract()?;
        let Some(generation_type) = d.get_item("type") else {
            return Err(py_required_key_error("type", "generation config"));
        };
        let generation_type: String = generation_type.extract()?;
        let generation_config = match generation_type.as_str() {
            "input_and_target" => {
                let Some(separator) = d.get_item("separator") else {
                    return Err(py_required_key_error(
                        "separator",
                        "input and target generation config",
                    ));
                };
                GenerationConfig::InputAndTarget(separator.extract()?)
            }
            "target_only" => GenerationConfig::TargetOnly,
            k => {
                return Err(py_invalid_type_error(k, "generation"));
            }
        };
        Ok(generation_config)
    }
}

#[derive(Debug, Clone)]
pub enum LabelingConfig {
    // whitespace correction labels given processed and original sequence
    WhitespaceCorrection(bool, TokenizerConfig),
    // the tokenization of input and/or target sequence
    Generation(TokenizerConfig, GenerationConfig),
    // no labeling
    None,
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
                let Some(tokenizer_config) = d.get_item("tokenizer") else {
                    return Err(py_required_key_error(
                        "tokenizer",
                        "whitespace correction config",
                    ));
                };
                LabelingConfig::WhitespaceCorrection(use_graphemes, tokenizer_config.extract()?)
            }
            "generation" => {
                let Some(tokenizer_config) = d.get_item("tokenizer") else {
                    return Err(py_required_key_error("tokenizer", "generation config"));
                };
                let Some(generation_config) = d.get_item("generation") else {
                    return Err(py_required_key_error("generation", "generation config"));
                };
                LabelingConfig::Generation(
                    tokenizer_config.extract()?,
                    generation_config.extract()?,
                )
            }
            "none" => LabelingConfig::None,
            k => {
                return Err(py_invalid_type_error(k, "labeling"));
            }
        };
        Ok(labeling_config)
    }
}

fn whitespace_correction_label(
    use_graphemes: bool,
    tokenizer_cfg: TokenizerConfig,
) -> Box<LabelingFn> {
    let tokenizer = tokenizer(tokenizer_cfg)
        .expect("failed to create tokenizer for whitespace correction label function");
    let num_prefix_tokens = tokenizer.num_prefix_tokens();
    let num_suffix_tokens = tokenizer.num_suffix_tokens();
    Box::new(move |item| {
        Ok(Label::SequenceClassification(
            vec![-1; num_prefix_tokens]
                .into_iter()
                .chain(
                    operations(&item.input, &item.target, use_graphemes)?
                        .into_iter()
                        .map(|l| l as i32),
                )
                .chain(vec![-1; num_suffix_tokens])
                .collect(),
        ))
    })
}

fn sequence_generation(
    tokenizer_cfg: TokenizerConfig,
    generation_cfg: GenerationConfig,
) -> Box<LabelingFn> {
    let tokenizer = tokenizer(tokenizer_cfg)
        .expect("failed to create tokenizer for conditional generation label function");
    Box::new(move |item| {
        let tokenization = tokenizer.tokenize(
            &match &generation_cfg {
                GenerationConfig::InputAndTarget(sep) => {
                    format!("{}{}{}", item.input, sep, item.target)
                }
                GenerationConfig::TargetOnly => item.target.clone(),
            },
            item.language.as_deref(),
            None,
            None,
            false,
        )?;
        let prefix_length = match &generation_cfg {
            GenerationConfig::InputAndTarget(sep) => {
                let prefix_tokenization = tokenizer.tokenize(
                    format!("{}{}", item.input, sep).trim_end(),
                    item.language.as_deref(),
                    None,
                    None,
                    false,
                )?;
                Some(
                    prefix_tokenization
                        .token_ids
                        .len()
                        .saturating_sub(tokenizer.num_suffix_tokens()),
                )
            }
            _ => None,
        };
        let token_ids = tokenization
            .token_ids
            .into_iter()
            .map(|t| t as i32)
            .collect();
        Ok(Label::Generation(
            token_ids,
            tokenizer.pad_token_id(),
            prefix_length,
        ))
    })
}

pub fn labeling(labeling: LabelingConfig) -> Box<LabelingFn> {
    match labeling {
        LabelingConfig::WhitespaceCorrection(use_graphemes, tokenizer) => {
            whitespace_correction_label(use_graphemes, tokenizer)
        }
        LabelingConfig::Generation(tokenizer, generation) => {
            sequence_generation(tokenizer, generation)
        }
        LabelingConfig::None => Box::new(|_| Ok(Label::Empty)),
    }
}
