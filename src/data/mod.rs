use crate::tokenization::Tokenization;
use serde::{Deserialize, Serialize};

pub mod preprocessing;
pub mod loading;

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct TextData {
    original: String,
    processed: String,
}

#[derive(Clone, Debug)]
pub enum Label {
    Classification(usize),
    SeqClassification(Vec<usize>),
    Seq2Seq(Vec<usize>),
}

#[derive(Clone, Debug)]
pub struct Item {
    data: TextData,
    tokenization: Tokenization<u16>,
    label: Label,
}

#[derive(Clone, Debug)]
pub struct Batch {
    items: Vec<Item>,
}
