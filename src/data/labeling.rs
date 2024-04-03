use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::{
    tokenization::{
        tokenizer, Tokenization, TokenizationConstraint, TokenizationConstraintConfig, Tokenizer,
        TokenizerConfig,
    },
    utils::{py_invalid_type_error, py_required_key_error},
    whitespace::operations,
};

use super::{Label, TextData};

pub type LabelingFn = dyn Send + Sync + 'static + Fn(&TextData) -> anyhow::Result<Label>;

#[derive(Debug, Clone)]
pub enum GenerationConfig {
    JoinTokenize {
        separator: String,
        tokenizer: TokenizerConfig,
        constraint: Option<TokenizationConstraintConfig>,
    },
    TokenizeJoin {
        tokenizer: TokenizerConfig,
        input_constraint: Option<TokenizationConstraintConfig>,
        target_constraint: Option<TokenizationConstraintConfig>,
    },
    OnlyTarget {
        tokenizer: TokenizerConfig,
        constraint: Option<TokenizationConstraintConfig>,
    },
}

impl<'a> FromPyObject<'a> for GenerationConfig {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let d: &PyDict = ob.extract()?;
        let Some(generation_type) = d.get_item("type")? else {
            return Err(py_required_key_error("type", "generation config"));
        };
        let generation_type: String = generation_type.extract()?;
        let generation_config = match generation_type.as_str() {
            "join_tokenize" => {
                let Some(separator) = d.get_item("separator")? else {
                    return Err(py_required_key_error(
                        "separator",
                        "join-tokenize generation config",
                    ));
                };
                let Some(tokenizer) = d.get_item("tokenizer")? else {
                    return Err(py_required_key_error(
                        "tokenizer",
                        "join-tokenize generation config",
                    ));
                };
                let constraint = d
                    .get_item("constraint")?
                    .map(|item| item.extract())
                    .transpose()?;
                GenerationConfig::JoinTokenize {
                    separator: separator.extract()?,
                    tokenizer: tokenizer.extract()?,
                    constraint,
                }
            }
            "tokenize_join" => {
                let Some(input_tokenizer) = d.get_item("input_tokenizer")? else {
                    return Err(py_required_key_error(
                        "input_tokenizer",
                        "tokenize-join generation config",
                    ));
                };
                let input_constraint = d
                    .get_item("input_constraint")?
                    .map(|item| item.extract())
                    .transpose()?;
                let target_constraint = d
                    .get_item("output_constraint")?
                    .map(|item| item.extract())
                    .transpose()?;
                GenerationConfig::TokenizeJoin {
                    tokenizer: input_tokenizer.extract()?,
                    input_constraint,
                    target_constraint,
                }
            }
            "only_target" => {
                let constraint = d
                    .get_item("constraint")?
                    .map(|item| item.extract())
                    .transpose()?;
                let tokenizer = d
                    .get_item("tokenizer")?
                    .map(|item| item.extract())
                    .transpose()?
                    .ok_or_else(|| {
                        py_required_key_error("tokenizer", "only target generation config")
                    })?;
                GenerationConfig::OnlyTarget {
                    tokenizer,
                    constraint,
                }
            }
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
    Generation(GenerationConfig),
    // no labeling
    None,
}

impl<'a> FromPyObject<'a> for LabelingConfig {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let d: &PyDict = ob.extract()?;
        let Some(labeling_type) = d.get_item("type")? else {
            return Err(py_required_key_error("type", "labeling config"));
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
                LabelingConfig::WhitespaceCorrection(use_graphemes, tokenizer_config.extract()?)
            }
            "generation" => {
                let Some(generation_config) = d.get_item("generation")? else {
                    return Err(py_required_key_error("generation", "generation config"));
                };
                LabelingConfig::Generation(generation_config.extract()?)
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

fn tokenize(
    input: &str,
    tokenizer: &Tokenizer,
    constraint: Option<&TokenizationConstraint>,
) -> anyhow::Result<Tokenization> {
    if let Some(constraint) = constraint {
        tokenizer.tokenize_with_constraint(input, None, None, false, constraint)
    } else {
        tokenizer.tokenize(input, None, None, false)
    }
}

fn sequence_generation(generation_cfg: GenerationConfig) -> Box<LabelingFn> {
    match generation_cfg {
        GenerationConfig::JoinTokenize {
            separator,
            tokenizer: tokenizer_cfg,
            constraint,
        } => {
            let tokenizer = tokenizer(tokenizer_cfg)
                .expect("failed to create tokenizer in sequence generation");
            let constraint = constraint
                .map(TokenizationConstraint::from_config)
                .transpose()
                .expect("failed to create tokenization constraint in sequence generation");
            Box::new(move |item| {
                let full = format!("{}{}{}", item.input, separator, item.target);
                let tokenization = tokenize(&full, &tokenizer, constraint.as_ref())?;
                Ok(Label::Generation(
                    tokenization
                        .token_ids
                        .into_iter()
                        .map(|t| t as i32)
                        .collect(),
                    tokenizer.pad_token_id(),
                    Some(
                        tokenize(
                            &format!("{}{}", item.input, separator),
                            &tokenizer,
                            constraint.as_ref(),
                        )?
                        .token_ids
                        .len()
                        .saturating_sub(tokenizer.num_suffix_tokens()),
                    ),
                ))
            })
        }
        GenerationConfig::TokenizeJoin {
            tokenizer: tokenizer_cfg,
            input_constraint,
            target_constraint,
        } => {
            let tokenizer = tokenizer(tokenizer_cfg)
                .expect("failed to create tokenizer in sequence generation");
            let input_constraint = input_constraint
                .map(TokenizationConstraint::from_config)
                .transpose()
                .expect("failed to create input tokenization constraint in sequence generation");
            let target_constraint = target_constraint
                .map(TokenizationConstraint::from_config)
                .transpose()
                .expect("failed to create target tokenization constraint in sequence generation");
            Box::new(move |item| {
                let input_tokenization =
                    tokenize(&item.input, &tokenizer, input_constraint.as_ref())?;
                let num_input_tokens = input_tokenization.token_ids.len();
                let target_tokenization =
                    tokenize(&item.target, &tokenizer, target_constraint.as_ref())?;
                let prefix_len = input_tokenization
                    .token_ids
                    .len()
                    .saturating_sub(tokenizer.num_suffix_tokens());
                Ok(Label::Generation(
                    input_tokenization
                        .token_ids
                        .into_iter()
                        .take(num_input_tokens - tokenizer.num_suffix_tokens())
                        .chain(
                            target_tokenization
                                .token_ids
                                .into_iter()
                                .skip(tokenizer.num_prefix_tokens()),
                        )
                        .map(|t| t as i32)
                        .collect(),
                    tokenizer.pad_token_id(),
                    Some(prefix_len),
                ))
            })
        }
        GenerationConfig::OnlyTarget {
            tokenizer: tokenizer_cfg,
            constraint,
        } => {
            let tokenizer = tokenizer(tokenizer_cfg)
                .expect("failed to create tokenizer in sequence generation");
            let constraint = constraint
                .map(TokenizationConstraint::from_config)
                .transpose()
                .expect("failed to create tokenization constraint in sequence generation");
            Box::new(move |item| {
                let tokenization = tokenize(&item.target, &tokenizer, constraint.as_ref())?;
                Ok(Label::Generation(
                    tokenization
                        .token_ids
                        .into_iter()
                        .map(|t| t as i32)
                        .collect(),
                    tokenizer.pad_token_id(),
                    None,
                ))
            })
        }
    }
}

pub fn labeling(labeling: LabelingConfig) -> Box<LabelingFn> {
    match labeling {
        LabelingConfig::WhitespaceCorrection(use_graphemes, tokenizer) => {
            whitespace_correction_label(use_graphemes, tokenizer)
        }
        LabelingConfig::Generation(generation) => sequence_generation(generation),
        LabelingConfig::None => Box::new(|_| Ok(Label::Empty)),
    }
}
