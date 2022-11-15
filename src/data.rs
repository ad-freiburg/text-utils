use std::ops::Sub;
use rand::{Rng, SeedableRng};
use rand_chacha::{ChaCha8Rng};
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

pub fn chain_preprocessing_fns(
    mut fs: Vec<Box<PreprocessingFn>>
) -> Box<PreprocessingFn> {
    Box::new(
        move |mut item| {
            for f in fs.iter_mut() {
                let new_item = f(item);
                if new_item.is_none() {
                    return None;
                } else {
                    item = new_item.unwrap();
                }
            }
            Some(item)
        }
    )
}

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

fn get_switch_fn(mut fns: Vec<Box<PreprocessingFn>>, probs: Vec<f64>, seed: u64) -> Box<PreprocessingFn> {
    let num_fns = fns.len();
    assert!(num_fns > 0 && num_fns == probs.len());
    // generate cumulative probabilities
    let cum_p: Vec<f64> = accumulate(probs.into_iter());
    // probabilities should sum to 1
    assert!(cum_p.last().copied().unwrap().sub(1f64).abs() < 1e-5);

    let mut rng = ChaCha8Rng::seed_from_u64(seed);

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
        Preprocessing::Switch(fns, probs, seed) => get_switch_fn(
            fns
                .into_iter()
                .map(|f| get_preprocessing_fn(f))
                .collect(),
            probs,
            seed,
        ),
        // Preprocessing::NoiseWhitespaces(iw_p, dw_p, seed) => {}
        Preprocessing::NoWhitespaces => get_no_whitespace_fn(),
        Preprocessing::FullWhitespaces => get_full_whitespace_fn(),
        Preprocessing::LabelWhitespaceCorrection => get_label_whitespace_correction_fn()
    }
}
