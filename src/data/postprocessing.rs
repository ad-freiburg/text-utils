use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Geometric};

use crate::tokenization::{tokenizer, TokenizerConfig};
use crate::utils::{py_invalid_type_error, py_required_key_error};

use super::utils::{chain, on_mark, switch, switch_on_mark};
use super::{TextDataInfo, TrainItem, TrainTaskInput};

pub type PostprocessingFn = dyn Send
    + Sync
    + 'static
    + Fn(TrainItem, TextDataInfo) -> anyhow::Result<(TrainItem, TextDataInfo)>;

#[derive(Debug, Clone)]
pub enum PostprocessingFnConfig {
    None,
    Chain(Vec<PostprocessingFnConfig>),
    Switch(Vec<PostprocessingFnConfig>, Vec<f64>),
    TokenMasking(TokenizerConfig, f64, usize, f64, String),
    OnMark(String, String, Vec<PostprocessingFnConfig>),
    SwitchOnMark(String, Vec<String>, Vec<PostprocessingFnConfig>),
    ClipLength,
}

impl<'a> FromPyObject<'a> for PostprocessingFnConfig {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let d: &PyDict = ob.extract()?;
        let Some(postprocessing_type) = d.get_item("type")? else {
            return Err(py_required_key_error("type", "postprocessing fn config"));
        };
        let postprocessing_type: String = postprocessing_type.extract()?;
        let postprocessing_config = match postprocessing_type.as_str() {
            "none" => PostprocessingFnConfig::None,
            "chain" => {
                let Some(cfgs) = d.get_item("configs")? else {
                    return Err(py_required_key_error("configs", "chain config"));
                };
                PostprocessingFnConfig::Chain(cfgs.extract()?)
            }
            "switch" => {
                let Some(configs) = d.get_item("configs")? else {
                    return Err(py_required_key_error("configs", "switch config"));
                };
                let Some(probs) = d.get_item("probabilities")? else {
                    return Err(py_required_key_error("probabilities", "switch config"));
                };
                PostprocessingFnConfig::Switch(configs.extract()?, probs.extract()?)
            }
            "on_mark" => {
                let Some(key) = d.get_item("key")? else {
                    return Err(py_required_key_error("key", "on mark config"));
                };
                let Some(value) = d.get_item("value")? else {
                    return Err(py_required_key_error("value", "on mark config"));
                };
                let Some(cfgs) = d.get_item("configs")? else {
                    return Err(py_required_key_error("configs", "on mark config"));
                };
                PostprocessingFnConfig::OnMark(key.extract()?, value.extract()?, cfgs.extract()?)
            }
            "switch_on_mark" => {
                let Some(key) = d.get_item("key")? else {
                    return Err(py_required_key_error("key", "switch on mark config"));
                };
                let Some(values) = d.get_item("values")? else {
                    return Err(py_required_key_error("values", "switch on mark config"));
                };
                let Some(configs) = d.get_item("configs")? else {
                    return Err(py_required_key_error("configs", "switch on mark config"));
                };
                PostprocessingFnConfig::SwitchOnMark(
                    key.extract()?,
                    values.extract()?,
                    configs.extract()?,
                )
            }
            "token_masking" => {
                let Some(tokenizer_cfg) = d.get_item("tokenizer")? else {
                    return Err(py_required_key_error("tokenizer", "token masking config"));
                };
                let Some(p) = d.get_item("prob")? else {
                    return Err(py_required_key_error("prob", "token masking config"));
                };
                let Some(min_tokens) = d.get_item("min_tokens")? else {
                    return Err(py_required_key_error("min_tokens", "token masking config"));
                };
                let Some(num_p) = d.get_item("num_tokens_prob")? else {
                    return Err(py_required_key_error(
                        "num_tokens_prob",
                        "token masking config",
                    ));
                };
                let Some(mask_token) = d.get_item("mask_token")? else {
                    return Err(py_required_key_error("mask_token", "token masking config"));
                };
                PostprocessingFnConfig::TokenMasking(
                    tokenizer_cfg.extract()?,
                    p.extract()?,
                    min_tokens.extract()?,
                    num_p.extract()?,
                    mask_token.extract()?,
                )
            }
            "clip_length" => PostprocessingFnConfig::ClipLength,
            k => {
                return Err(py_invalid_type_error(k, "postprocessing fn"));
            }
        };
        Ok(postprocessing_config)
    }
}

fn mask_tokens(
    p: f64,
    min: usize,
    num_p: f64,
    mask_token_id: u32,
    num_pfx_tokens: usize,
    num_sfx_tokens: usize,
) -> Box<PostprocessingFn> {
    // adjust num_p because rusts geometric distribution is defined as
    // the number of failures until first success, but we want the number of
    // trails including the first success (also including 0)
    let num_p = 1.0 / (1.0 / num_p + 1.0);
    let expected_per_mask = min as f64 + 1.0 / num_p;
    let geo = Geometric::new(p).expect("failed to create geometric distribution");
    assert!(min > 0, "minimum tokens to mask must be greater than 0");
    // adjust mask probability because it refers to the percentage of tokens expected to be masked
    // but we always mask multiple tokens per mask operation
    let p = p / expected_per_mask;
    Box::new(move |mut item, info| {
        let mut rng = ChaCha8Rng::seed_from_u64(info.seed);

        let token_ids = match &mut item.input {
            TrainTaskInput::Classification { token_ids, .. } => token_ids.as_mut_slice(),
            TrainTaskInput::SequenceClassification { token_ids, .. } => token_ids.as_mut_slice(),
            TrainTaskInput::Generation { token_ids, .. } => token_ids.as_mut_slice(),
            TrainTaskInput::ConditionalGeneration { token_ids, .. } => token_ids.as_mut_slice(),
        };
        if token_ids.len() <= 1 {
            return Ok((item, info));
        }
        let mut i = 0;
        let num_maskable_tokens = token_ids.len() - num_pfx_tokens - num_sfx_tokens;
        while i < num_maskable_tokens {
            if rng.gen::<f64>() > p {
                i += 1;
                continue;
            }
            // a single masking should never be more than half of the tokens
            // to allow for more context between the tokens, should only happen
            // for very short sequences or very high corruption values
            let num_to_mask = (geo.sample(&mut rng) as usize + min)
                .min(num_maskable_tokens / 2)
                .min(num_maskable_tokens - i);
            token_ids[i + num_pfx_tokens..i + num_pfx_tokens + num_to_mask]
                .iter_mut()
                .for_each(|token_id| *token_id = mask_token_id);
            // one mask token for each masked character
            i += num_to_mask;
        }
        Ok((item, info))
    })
}

#[inline]
fn clip_input(mut input: TrainTaskInput, length: usize) -> TrainTaskInput {
    match input {
        TrainTaskInput::Classification {
            ref mut token_ids, ..
        } => {
            token_ids.truncate(length);
        }
        TrainTaskInput::SequenceClassification {
            ref mut token_ids,
            ref mut labels,
            ..
        } => {
            token_ids.truncate(length);
            labels.truncate(length);
        }
        TrainTaskInput::Generation {
            ref mut token_ids,
            ref mut labels,
            ..
        } => {
            token_ids.truncate(length);
            labels.truncate(length);
        }
        TrainTaskInput::ConditionalGeneration {
            ref mut token_ids,
            ref mut target_token_ids,
            ref mut labels,
            ..
        } => {
            token_ids.truncate(length);
            target_token_ids.truncate(length);
            labels.truncate(length);
        }
    };
    input
}

pub fn clip_length(max_length: Arc<AtomicUsize>) -> Box<PostprocessingFn> {
    Box::new(move |item, info| {
        let length = max_length.load(Ordering::Relaxed);
        let input = clip_input(item.input, length);
        Ok((TrainItem { input, ..item }, info))
    })
}

pub fn postprocessing(
    cfg: PostprocessingFnConfig,
    max_length: Arc<AtomicUsize>,
) -> Box<PostprocessingFn> {
    match cfg {
        PostprocessingFnConfig::None => Box::new(|item, info| Ok((item, info))),
        PostprocessingFnConfig::Chain(configs) => {
            let pfns = configs
                .into_iter()
                .map(|cfg| postprocessing(cfg, max_length.clone()))
                .collect();
            chain(pfns)
        }
        PostprocessingFnConfig::Switch(configs, probs) => {
            let pfns = configs
                .into_iter()
                .map(|cfg| postprocessing(cfg, max_length.clone()))
                .collect();
            switch(pfns, probs)
        }
        PostprocessingFnConfig::OnMark(key, value, configs) => {
            let pfn = postprocessing(PostprocessingFnConfig::Chain(configs), max_length);
            on_mark(pfn, key, value)
        }
        PostprocessingFnConfig::SwitchOnMark(key, values, configs) => {
            let pfns = configs
                .into_iter()
                .map(|cfg| postprocessing(cfg, max_length.clone()))
                .collect();
            switch_on_mark(pfns, key, values)
        }
        PostprocessingFnConfig::ClipLength => clip_length(max_length),
        PostprocessingFnConfig::TokenMasking(tokenizer_cfg, p, min, num_p, mask_token) => {
            let Ok(tokenizer) = tokenizer(tokenizer_cfg) else {
                panic!("failed to create tokenizer for token masking");
            };
            let Some(mask_token_id) = tokenizer.special_token_to_id(&mask_token) else {
                panic!("mask token {mask_token} not found in tokenizer");
            };
            mask_tokens(
                p,
                min,
                num_p,
                mask_token_id,
                tokenizer.num_prefix_tokens(),
                tokenizer.num_suffix_tokens(),
            )
        }
    }
}
