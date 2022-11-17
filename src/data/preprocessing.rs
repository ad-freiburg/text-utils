use std::fs::read_to_string;
use std::ops::Sub;
use std::path::Path;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use crate::data::{TextData, Label};
use crate::utils::accumulate;
use crate::whitespace::{full, operations, remove};

pub type PreprocessingFn = Box<dyn FnMut(TextData) -> TextData>;
pub type LabelingFn = Box<dyn FnMut(&TextData) -> Label>;

#[derive(Serialize, Deserialize, PartialEq, Debug)]
pub enum Preprocessing {
    // switch between multiple preprocessing functions
    Switch(Vec<Preprocessing>, Vec<f64>, u64),
    // delete all whitespaces
    NoWhitespaces,
    // insert whitespaces between all characters
    FullWhitespaces,
    // delete and insert whitespaces with certain probabilities
    // NoiseWhitespaces(f64, f64, u64),
}

#[derive(Serialize, Deserialize, PartialEq, Debug)]
pub enum Labeling {
    // generate whitespace correction labels given processed and original sequence
    LabelWhitespaceCorrection,
}

fn switch(fns: Vec<Preprocessing>, probs: Vec<f64>, seed: u64) -> PreprocessingFn {
    let num_fns = fns.len();
    assert!(num_fns > 0 && num_fns == probs.len());
    // generate cumulative probabilities
    let cum_p: Vec<f64> = accumulate(&probs);
    // probabilities should sum to 1
    assert!(cum_p.last().copied().unwrap().sub(1f64).abs() < 1e-5);

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut fns: Vec<PreprocessingFn> = fns
        .into_iter()
        .map(|f| preprocessing_fn(f))
        .collect();

    // return new function that switches between multiple preprocessing functions
    // based on the given probability distribution
    Box::new(
        move |item| {
            let r: f64 = rng.gen();
            let mut idx = 0;
            while idx < num_fns - 1 && r > cum_p[idx] {
                idx += 1;
            }
            fns[idx](item)
        }
    )
}

fn apply_to_text(f: fn(&str) -> String) -> PreprocessingFn {
    Box::new(
        move |item| {
            TextData { processed: f(&item.processed), ..item }
        }
    )
}

fn no_whitespace() -> PreprocessingFn {
    apply_to_text(remove)
}

fn full_whitespace() -> PreprocessingFn {
    apply_to_text(full)
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

fn preprocessing_fn(preprocessing: Preprocessing) -> PreprocessingFn {
    match preprocessing {
        Preprocessing::Switch(fns, probs, seed) => switch(fns, probs, seed),
        // Preprocessing::NoiseWhitespaces(iw_p, dw_p, seed) => {}
        Preprocessing::NoWhitespaces => no_whitespace(),
        Preprocessing::FullWhitespaces => full_whitespace(),
    }
}

pub fn preprocessing(
    preprocessing: Vec<Preprocessing>
) -> PreprocessingFn {
    // return new function that runs all given preprocessing functions
    // in order
    let mut fns: Vec<PreprocessingFn> = preprocessing
        .into_iter()
        .map(|p| preprocessing_fn(p))
        .collect();
    Box::new(
        move |mut item| {
            for f in fns.iter_mut() {
                item = f(item);
            }
            item
        }
    )
}

pub fn preprocessing_from_yaml(path: &Path) -> PreprocessingFn {
    let raw_yaml = read_to_string(path)
        .expect(&format!("could not read yaml file at {:?}", path));
    preprocessing_from_str(&raw_yaml)
}

pub fn preprocessing_from_str(s: &str) -> PreprocessingFn {
    let fns: Vec<Preprocessing> = serde_yaml::from_str(s)
        .expect(&format!("could not deserialize from yaml string\n{}", s));
    preprocessing(fns)
}

pub fn labeling(labeling: Labeling) -> LabelingFn {
    match labeling {
        Labeling::LabelWhitespaceCorrection => whitespace_correction_label()
    }
}
