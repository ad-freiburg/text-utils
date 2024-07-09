use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::{
    tokenization::{
        tokenizer, TokenizationConstraint, TokenizationConstraintConfig, TokenizerConfig,
    },
    utils::{py_invalid_type_error, py_required_key_error},
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
    // constrained text generation
    ConstrainedGeneration(
        bool,
        TokenizerConfig,
        bool,
        TokenizationConstraintConfig,
        Option<String>,
        Option<String>,
    ),
    // conditional generation aka text-to-text
    ConditionalGeneration(TokenizerConfig, bool, TokenizerConfig, bool),
}

impl<'a> FromPyObject<'a> for TrainTaskConfig {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let d: &PyDict = ob.extract()?;
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
            "constrained_generation" => {
                let mask_input = d
                    .get_item("mask_input")?
                    .map(|item| item.extract())
                    .transpose()?
                    .unwrap_or_default();
                let Some(tokenizer_config) = d.get_item("tokenizer")? else {
                    return Err(py_required_key_error(
                        "tokenizer",
                        "constrained generation config",
                    ));
                };
                let ignore_special_tokens = d
                    .get_item("ignore_special_tokens")?
                    .map(|item| item.extract())
                    .transpose()?
                    .unwrap_or_default();
                let Some(constraint_config) = d.get_item("constraint")? else {
                    return Err(py_required_key_error(
                        "constraint",
                        "constrained generation config",
                    ));
                };
                let separator = d
                    .get_item("separator")?
                    .map(|item| item.extract())
                    .transpose()?;
                let suffix = d
                    .get_item("suffix")?
                    .map(|item| item.extract())
                    .transpose()?;
                TrainTaskConfig::ConstrainedGeneration(
                    mask_input,
                    tokenizer_config.extract()?,
                    ignore_special_tokens,
                    constraint_config.extract()?,
                    separator,
                    suffix,
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
    constraint: Option<TokenizationConstraintConfig>,
    separator: Option<String>,
    suffix: Option<String>,
) -> Box<TaskFn> {
    let tokenizer =
        tokenizer(tokenizer_cfg).expect("failed to create tokenizer for generation input function");
    let constraint = constraint
        .map(TokenizationConstraint::from_config)
        .transpose()
        .expect("failed to create tokenization constraint for generation input function");
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
        let mut token_ids = if let Some(constraint) = &constraint {
            let mut token_ids = tokenizer
                .tokenize(
                    &format!("{}{}", item.input, separator.as_deref().unwrap_or_default()),
                    ignore_special_tokens,
                )?
                .token_ids;
            if let Err(e) =
                tokenizer.tokenize_with_constraint(&item.target, ignore_special_tokens, constraint)
            {
                println!("Failed to tokenize with constraint: {e}");
            };
            let constrained_token_ids = tokenizer
                .tokenize_with_constraint(&item.target, ignore_special_tokens, constraint)?
                .token_ids;
            token_ids.extend(constrained_token_ids);
            if let Some(suffix) = suffix.as_ref() {
                token_ids.extend(tokenizer.tokenize(suffix, ignore_special_tokens)?.token_ids);
            }
            token_ids
        } else {
            let joined = format!(
                "{}{}{}{}",
                item.input,
                separator.as_deref().unwrap_or_default(),
                item.target,
                suffix.as_deref().unwrap_or_default()
            );
            tokenizer
                .tokenize(&joined, ignore_special_tokens)?
                .token_ids
        };
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
            None,
            separator,
            None,
        ),
        TrainTaskConfig::ConstrainedGeneration(
            mask_input,
            tokenizer_cfg,
            ignore_special_tokens,
            constraint,
            separator,
            suffix,
        ) => generation_input(
            mask_input,
            tokenizer_cfg,
            ignore_special_tokens,
            Some(constraint),
            separator,
            suffix,
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
    }
}
