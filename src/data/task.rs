use anyhow::anyhow;
use std::collections::HashMap;

use itertools::Itertools;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::{
    tokenization::{tokenizer, TokenizerConfig},
    utils::{py_invalid_type_error, py_required_key_error, py_value_error},
    whitespace::operations,
};

use super::{TrainData, TrainTaskInput};

pub type TaskFn = dyn Send + Sync + 'static + Fn(&TrainData) -> anyhow::Result<TrainTaskInput>;

#[derive(Debug, Clone)]
pub enum TrainTaskConfig {
    // whitespace correction labels given processed and original sequence
    WhitespaceCorrection(bool, TokenizerConfig),
    // text generation aka language modeling
    Generation(bool, TokenizerConfig, bool, Option<String>),
    // conditional generation aka text-to-text
    ConditionalGeneration(TokenizerConfig, bool, TokenizerConfig, bool),
    // classification
    Classification(TokenizerConfig, bool, Vec<String>),
}

impl<'a> FromPyObject<'a> for TrainTaskConfig {
    fn extract_bound(ob: &Bound<'a, PyAny>) -> PyResult<Self> {
        let d: &Bound<'_, PyDict> = ob.downcast()?;
        let Some(labeling_type) = d.get_item("type")? else {
            return Err(py_required_key_error("type", "task config"));
        };
        let labeling_type: String = labeling_type.extract()?;
        let labeling_config = match labeling_type.as_str() {
            "whitespace_correction" => {
                let use_graphemes = if let Some(value) = d.get_item("use_graphemes")? {
                    value.extract()?
                } else {
                    true
                };
                let Some(tokenizer_config) = d.get_item("tokenizer")? else {
                    return Err(py_required_key_error(
                        "tokenizer",
                        "whitespace correction config",
                    ));
                };
                TrainTaskConfig::WhitespaceCorrection(use_graphemes, tokenizer_config.extract()?)
            }
            "generation" => {
                let mask_input = d
                    .get_item("mask_input")?
                    .map(|item| item.extract())
                    .transpose()?
                    .unwrap_or_default();
                let Some(tokenizer_config) = d.get_item("tokenizer")? else {
                    return Err(py_required_key_error("tokenizer", "generation config"));
                };
                let ignore_special_tokens = d
                    .get_item("ignore_special_tokens")?
                    .map(|item| item.extract())
                    .transpose()?
                    .unwrap_or_default();
                let separator = d
                    .get_item("separator")?
                    .map(|item| item.extract())
                    .transpose()?;
                TrainTaskConfig::Generation(
                    mask_input,
                    tokenizer_config.extract()?,
                    ignore_special_tokens,
                    separator,
                )
            }
            "conditional_generation" => {
                let Some(input_tokenizer) = d.get_item("input_tokenizer")? else {
                    return Err(py_required_key_error(
                        "input_tokenizer",
                        "conditional generation config",
                    ));
                };
                let input_ignore_special_tokens = d
                    .get_item("input_ignore_special_tokens")?
                    .map(|item| item.extract())
                    .transpose()?
                    .unwrap_or_default();
                let Some(target_tokenizer) = d.get_item("target_tokenizer")? else {
                    return Err(py_required_key_error(
                        "target_tokenizer",
                        "conditional generation config",
                    ));
                };
                let target_ignore_special_tokens = d
                    .get_item("target_ignore_special_tokens")?
                    .map(|item| item.extract())
                    .transpose()?
                    .unwrap_or_default();
                TrainTaskConfig::ConditionalGeneration(
                    input_tokenizer.extract()?,
                    input_ignore_special_tokens,
                    target_tokenizer.extract()?,
                    target_ignore_special_tokens,
                )
            }
            "classification" => {
                let Some(tokenizer_config) = d.get_item("tokenizer")? else {
                    return Err(py_required_key_error("tokenizer", "classification config"));
                };
                let ignore_special_tokens = d
                    .get_item("ignore_special_tokens")?
                    .map(|item| item.extract())
                    .transpose()?
                    .unwrap_or_default();
                let Some(classes) = d.get_item("classes")? else {
                    return Err(py_required_key_error("classes", "classification config"));
                };
                let classes: Vec<_> = classes.extract()?;
                if classes.len() < 2 {
                    return Err(py_value_error(
                        "classification requires at least two classes",
                    ));
                } else if classes.iter().unique().count() != classes.len() {
                    return Err(py_value_error("classes must be unique"));
                }
                TrainTaskConfig::Classification(
                    tokenizer_config.extract()?,
                    ignore_special_tokens,
                    classes,
                )
            }
            k => {
                return Err(py_invalid_type_error(k, "task"));
            }
        };
        Ok(labeling_config)
    }
}

fn whitespace_correction_input(use_graphemes: bool, tokenizer_cfg: TokenizerConfig) -> Box<TaskFn> {
    let tokenizer = tokenizer(tokenizer_cfg)
        .expect("failed to create tokenizer for whitespace correction input function");
    let num_prefix_tokens = tokenizer.num_prefix_tokens();
    let num_suffix_tokens = tokenizer.num_suffix_tokens();
    Box::new(move |item| {
        Ok(TrainTaskInput::SequenceClassification {
            token_ids: tokenizer.tokenize(&item.input, true)?.token_ids,
            pad_token_id: tokenizer.pad_token_id(),
            labels: vec![-1; num_prefix_tokens]
                .into_iter()
                .chain(
                    operations(&item.input, &item.target, use_graphemes)?
                        .into_iter()
                        .map(|l| l as i32),
                )
                .chain(vec![-1; num_suffix_tokens])
                .collect(),
        })
    })
}

fn generation_input(
    mask_prefix: bool,
    tokenizer_cfg: TokenizerConfig,
    ignore_special_tokens: bool,
    separator: Option<String>,
    suffix: Option<String>,
) -> Box<TaskFn> {
    let tokenizer =
        tokenizer(tokenizer_cfg).expect("failed to create tokenizer for generation input function");
    Box::new(move |item| {
        let mask_len = if mask_prefix {
            let pfx = format!("{}{}", item.input, separator.as_deref().unwrap_or_default());
            tokenizer
                .tokenize(&pfx, ignore_special_tokens)?
                .token_ids
                .len()
                - tokenizer.num_suffix_tokens()
        } else {
            0
        };
        let joined = format!(
            "{}{}{}{}",
            item.input,
            separator.as_deref().unwrap_or_default(),
            item.target,
            suffix.as_deref().unwrap_or_default()
        );
        let mut token_ids = tokenizer
            .tokenize(&joined, ignore_special_tokens)?
            .token_ids;
        // for n tokens, 1..n-1 are input, 2..n are labels
        let labels = vec![-1; mask_len]
            .into_iter()
            .chain(token_ids.iter().skip(mask_len).map(|l| *l as i32))
            .skip(1)
            .collect();
        token_ids.pop();
        Ok(TrainTaskInput::Generation {
            token_ids,
            pad_token_id: tokenizer.pad_token_id(),
            labels,
        })
    })
}

fn conditional_generation_input(
    input_tokenizer_cfg: TokenizerConfig,
    input_ignore_special_tokens: bool,
    target_tokenizer_cfg: TokenizerConfig,
    target_ignore_special_tokens: bool,
) -> Box<TaskFn> {
    let input_tokenizer = tokenizer(input_tokenizer_cfg)
        .expect("failed to create input tokenizer for conditional generation input function");
    let target_tokenizer = tokenizer(target_tokenizer_cfg)
        .expect("failed to create target tokenizer for conditional generation input function");
    Box::new(move |item| {
        let token_ids = input_tokenizer
            .tokenize(&item.input, input_ignore_special_tokens)?
            .token_ids;
        let mut target_token_ids = target_tokenizer
            .tokenize(&item.target, target_ignore_special_tokens)?
            .token_ids;
        let labels = target_token_ids.iter().map(|l| *l as i32).skip(1).collect();
        target_token_ids.pop();
        Ok(TrainTaskInput::ConditionalGeneration {
            token_ids,
            pad_token_id: input_tokenizer.pad_token_id(),
            target_token_ids,
            target_pad_token_id: target_tokenizer.pad_token_id(),
            labels,
        })
    })
}

fn classification_input(
    tokenizer_cfg: TokenizerConfig,
    ignore_special_tokens: bool,
    classes: Vec<String>,
) -> Box<TaskFn> {
    assert!(
        classes.len() <= i32::MAX as usize,
        "too many classes for classification task"
    );
    let tokenizer = tokenizer(tokenizer_cfg)
        .expect("failed to create tokenizer for classification input function");
    let class_to_index: HashMap<_, _> = classes
        .into_iter()
        .enumerate()
        .map(|(i, c)| (c, i as i32))
        .collect();
    Box::new(move |item| {
        Ok(TrainTaskInput::Classification {
            token_ids: tokenizer
                .tokenize(&item.input, ignore_special_tokens)?
                .token_ids,
            pad_token_id: tokenizer.pad_token_id(),
            label: class_to_index.get(&item.target).copied().ok_or_else(|| {
                anyhow!("class '{}' not found in classification task", item.target)
            })?,
        })
    })
}

pub fn train_task(task: TrainTaskConfig) -> Box<TaskFn> {
    match task {
        TrainTaskConfig::WhitespaceCorrection(use_graphemes, tokenizer) => {
            whitespace_correction_input(use_graphemes, tokenizer)
        }
        TrainTaskConfig::Generation(
            mask_input,
            tokenizer_cfg,
            ignore_special_tokens,
            separator,
        ) => generation_input(
            mask_input,
            tokenizer_cfg,
            ignore_special_tokens,
            separator,
            None,
        ),
        TrainTaskConfig::ConditionalGeneration(
            input_tokenizer,
            input_ignore_special_tokens,
            target_tokenizer,
            target_ignore_special_tokens,
        ) => conditional_generation_input(
            input_tokenizer,
            input_ignore_special_tokens,
            target_tokenizer,
            target_ignore_special_tokens,
        ),
        TrainTaskConfig::Classification(tokenizer_config, ignore_special_tokens, classes) => {
            classification_input(tokenizer_config, ignore_special_tokens, classes)
        }
    }
}
