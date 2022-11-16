use std::fs::read_to_string;
use std::ops::Sub;
use std::path::Path;
use rand::{Rng, SeedableRng};
use rand_chacha::{ChaCha8Rng};
use serde::{Deserialize, Serialize};
use anyhow::Result;
use crate::utils::accumulate;
use crate::whitespace::{full, operations, remove};

pub enum Label {
    SeqClassification(Vec<usize>),
    Seq2Seq(String),
}

pub enum Item {
    Input(String),
    InputAndTarget(String, String),
    InputAndLabel(String, Label),
}

pub type PreprocessingFn = dyn FnMut(Item) -> Option<Item>;

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
    // generate whitespace correction labels given input and target sequence
    LabelWhitespaceCorrection,
}

fn get_switch_fn(fns: Vec<Preprocessing>, probs: Vec<f64>, seed: u64) -> Box<PreprocessingFn> {
    let num_fns = fns.len();
    assert!(num_fns > 0 && num_fns == probs.len());
    // generate cumulative probabilities
    let cum_p: Vec<f64> = accumulate(probs.iter());
    // probabilities should sum to 1
    assert!(cum_p.last().copied().unwrap().sub(1f64).abs() < 1e-5);

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut fns: Vec<Box<PreprocessingFn>> = fns
        .into_iter()
        .map(|f| get_preprocessing_fn(f))
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

fn get_apply_to_text_fn(f: fn(&str) -> String) -> Box<PreprocessingFn> {
    Box::new(
        move |item| {
            match item {
                Item::Input(text) => {
                    Some(Item::InputAndTarget(f(&text), text))
                }
                Item::InputAndTarget(text, target) => {
                    Some(Item::InputAndTarget(f(&text), target))
                }
                _ => panic!("input should be a text only or a text and target item")
            }
        }
    )
}

fn get_no_whitespace_fn() -> Box<PreprocessingFn> {
    get_apply_to_text_fn(remove)
}

fn get_full_whitespace_fn() -> Box<PreprocessingFn> {
    get_apply_to_text_fn(full)
}

fn get_label_whitespace_correction_fn() -> Box<PreprocessingFn> {
    Box::new(
        |item| {
            match item {
                Item::InputAndTarget(from, to) => {
                    let labels = operations(&from, &to);
                    Some(Item::InputAndLabel(from, Label::SeqClassification(labels)))
                }
                _ => panic!("label whitespace correction requires an item with input and target text")
            }
        }
    )
}

pub fn get_preprocessing_fn(preprocessing: Preprocessing) -> Box<PreprocessingFn> {
    match preprocessing {
        Preprocessing::Switch(fns, probs, seed) => get_switch_fn(fns, probs, seed),
        // Preprocessing::NoiseWhitespaces(iw_p, dw_p, seed) => {}
        Preprocessing::NoWhitespaces => get_no_whitespace_fn(),
        Preprocessing::FullWhitespaces => get_full_whitespace_fn(),
        Preprocessing::LabelWhitespaceCorrection => get_label_whitespace_correction_fn()
    }
}

pub fn get_preprocessing_fns(
    preprocessing: Vec<Preprocessing>
) -> Box<PreprocessingFn> {
    // return new function that runs all given preprocessing functions
    // in order
    let mut fns: Vec<Box<PreprocessingFn>> = preprocessing
        .into_iter()
        .map(|p| get_preprocessing_fn(p))
        .collect();
    Box::new(
        move |mut item| {
            for f in fns.iter_mut() {
                if let Some(new_item) = f(item) {
                    item = new_item;
                } else {
                    return None;
                }
            }
            Some(item)
        }
    )
}

pub fn get_preprocessing_fn_from_config(path: &Path) -> Result<Box<PreprocessingFn>> {
    let raw_yaml = read_to_string(path)?;
    let fns: Vec<Preprocessing> = serde_yaml::from_str(&raw_yaml)?;
    Ok(get_preprocessing_fns(fns))
}
