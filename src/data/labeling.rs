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
pub enum LabelingConfig {
    // whitespace correction labels given processed and original sequence
    WhitespaceCorrection(bool, TokenizerConfig),
    // the tokenization of the original sequence
    ConditionalGeneration(TokenizerConfig),
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
                    return Err(py_required_key_error("tokenizer", "whitespace correction config"));
                };
                LabelingConfig::WhitespaceCorrection(use_graphemes, tokenizer_config.extract()?)
            }
            "conditional_generation" => {
                let Some(tokenizer_config) = d.get_item("tokenizer") else {
                    return Err(py_required_key_error("tokenizer", "conditional generation config"));
                };
                LabelingConfig::ConditionalGeneration(tokenizer_config.extract()?)
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

fn conditional_sequence_generation(tokenizer_cfg: TokenizerConfig) -> Box<LabelingFn> {
    let tokenizer = tokenizer(tokenizer_cfg)
        .expect("failed to create tokenizer for conditional generation label function");
    Box::new(move |item| {
        let tokenization =
            tokenizer.tokenize(&item.target, item.language.as_deref(), None, None, false)?;
        let token_ids = tokenization
            .token_ids
            .into_iter()
            .map(|t| t as i32)
            .collect();
        Ok(Label::ConditionalGeneration(
            token_ids,
            tokenizer.pad_token_id(),
        ))
    })
}

pub fn labeling(labeling: LabelingConfig) -> Box<LabelingFn> {
    match labeling {
        LabelingConfig::WhitespaceCorrection(use_graphemes, tokenizer) => {
            whitespace_correction_label(use_graphemes, tokenizer)
        }
        LabelingConfig::ConditionalGeneration(tokenizer) => {
            conditional_sequence_generation(tokenizer)
        }
        LabelingConfig::None => Box::new(|_| Ok(Label::Empty)),
    }
}
