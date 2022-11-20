use std::ops::Sub;
use itertools::Itertools;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use crate::data::{TextData, Label};
use crate::text;
use crate::unicode::CS;
use crate::utils::accumulate;
use crate::whitespace::{full, operations, remove};

pub type PreprocessingFn = Box<dyn Fn(TextData, Option<u64>) -> TextData + Send + Sync>;
pub type LabelingFn = Box<dyn Fn(&TextData) -> Label + Send + Sync>;

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub enum PreprocessingConfig {
    // clean the sequences (remove spurious whitespaces)
    Clean,
    // switch between multiple preprocessing functions
    Switch(Vec<PreprocessingConfig>, Vec<f64>),
    // delete all whitespaces
    NoWhitespaces,
    // insert whitespaces between all characters
    FullWhitespaces,
    // delete and insert whitespaces with certain probabilities
    NoiseWhitespaces(f64, f64),
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub enum LabelingConfig {
    // generate whitespace correction labels given processed and original sequence
    LabelWhitespaceCorrection,
}

fn switch(fns: Vec<PreprocessingConfig>, probs: Vec<f64>) -> PreprocessingFn {
    let num_fns = fns.len();
    assert!(num_fns > 0 && num_fns == probs.len());
    // generate cumulative probabilities
    let cum_p: Vec<f64> = accumulate(&probs);
    // probabilities should sum to 1
    assert!(cum_p.last().copied().unwrap().sub(1f64).abs() < 1e-5);

    let fns: Vec<PreprocessingFn> = fns
        .into_iter()
        .map(|f| preprocessing_fn(f))
        .collect();

    // return new function that switches between multiple preprocessing functions
    // based on the given probability distribution
    Box::new(
        move |item, seed| {
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
        }
    )
}

fn apply_to_text<F: Fn(&str) -> String + Send + Sync + 'static>(f: F) -> PreprocessingFn {
    Box::new(
        move |item, _| {
            TextData { processed: f(&item.processed), ..item }
        }
    )
}

fn noise_whitespace(iw_p: f64, dw_p: f64) -> PreprocessingFn {
    Box::new(
        move |item, seed| {
            let mut rng = if seed.is_some() {
                ChaCha8Rng::seed_from_u64(seed.unwrap())
            } else {
                ChaCha8Rng::from_entropy()
            };
            let cs = CS::new(&item.processed, true);
            let processed = cs
                .chars()
                .map(|c| {
                    let r: f64 = rng.gen();
                    if c.is_whitespace() {
                        if r < dw_p {
                            "".to_string()
                        } else {
                            c.str.to_string()
                        }
                    } else if r < iw_p {
                        " ".to_string() + c.str
                    } else {
                        c.str.to_string()
                    }
                })
                .join("");
            TextData { processed, ..item }
        }
    )
}

fn preprocessing_fn(preprocessing: PreprocessingConfig) -> PreprocessingFn {
    match preprocessing {
        PreprocessingConfig::Clean => apply_to_text(text::clean),
        PreprocessingConfig::Switch(fns, probs) => switch(fns, probs),
        PreprocessingConfig::NoiseWhitespaces(iw_p, dw_p) => noise_whitespace(iw_p, dw_p),
        PreprocessingConfig::NoWhitespaces => apply_to_text(remove),
        PreprocessingConfig::FullWhitespaces => apply_to_text(full),
    }
}

pub fn preprocessing(
    preprocessing: Vec<PreprocessingConfig>
) -> PreprocessingFn {
    // return new function that runs all given preprocessing functions
    // in order
    let fns: Vec<PreprocessingFn> = preprocessing
        .into_iter()
        .map(|p| preprocessing_fn(p))
        .collect();
    Box::new(
        move |mut item, seed| {
            for f in fns.iter() {
                item = f(item, seed);
            }
            item
        }
    )
}

fn whitespace_correction_label() -> LabelingFn {
    Box::new(
        |item| {
            Label::SeqClassification(
                operations(&item.processed, &item.original)
            )
        }
    )
}

pub fn labeling(labeling: LabelingConfig) -> LabelingFn {
    match labeling {
        LabelingConfig::LabelWhitespaceCorrection => whitespace_correction_label()
    }
}
