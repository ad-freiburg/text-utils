use std::{collections::HashMap, ops::Sub};

use anyhow::anyhow;
use itertools::Itertools;
use pyo3::{prelude::*, types::PyDict};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use crate::utils::{accumulate, py_required_key_error};

use super::TextDataInfo;

pub type Function<I, O> =
    dyn Send + Sync + 'static + Fn(I, TextDataInfo) -> anyhow::Result<(O, TextDataInfo)>;

pub fn switch<I: 'static, O: 'static>(
    fns: Vec<Box<Function<I, O>>>,
    probs: Vec<f64>,
) -> Box<Function<I, O>> {
    let num_fns = fns.len();
    assert!(
        num_fns > 0 && num_fns == probs.len(),
        "expected one or more fns for switch fn and the same \
        number of probabilities"
    );
    // generate cumulative probabilities
    let cum_p: Vec<f64> = accumulate(&probs);
    // probabilities should sum to 1
    assert!(
        cum_p.last().copied().unwrap().sub(1f64).abs() < 1e-5,
        "all switch probabilities should sum to 1"
    );

    // return new function that switches between multiple preprocessing functions
    // based on the given probability distribution
    Box::new(
        move |input: I, info: TextDataInfo| -> anyhow::Result<(O, TextDataInfo)> {
            let mut rng = ChaCha8Rng::seed_from_u64(info.seed);
            let r: f64 = rng.gen();
            let mut idx = 0;
            while idx < num_fns - 1 && r > cum_p[idx] {
                idx += 1;
            }
            fns[idx](input, info)
        },
    )
}

pub fn switch_on_mark<I: 'static, O: 'static>(
    fns: Vec<Box<Function<I, O>>>,
    key: String,
    values: Vec<String>,
) -> Box<Function<I, O>> {
    let num_fns = fns.len();
    assert!(
        num_fns > 0 && num_fns == values.len(),
        "expected one or more fns for the switch fn and the same \
        number of mark values"
    );
    // marks should be unique
    assert!(
        values.iter().unique().count() == values.len(),
        "all mark values should be unique"
    );

    Box::new(
        move |input: I, info: TextDataInfo| -> anyhow::Result<(O, TextDataInfo)> {
            let mark = info
                .marks
                .get(&key)
                .unwrap_or_else(|| panic!("mark \"{key}\" not found"));
            let idx = values
                .iter()
                .position(|v| v == mark)
                .unwrap_or_else(|| panic!("\"{mark}\" not found in supported values"));
            fns[idx](input, info)
        },
    )
}

pub fn chain<I: 'static>(fns: Vec<Box<Function<I, I>>>) -> Box<Function<I, I>> {
    Box::new(
        move |mut input: I, mut info| -> anyhow::Result<(I, TextDataInfo)> {
            for f in &fns {
                let output = f(input, info)?;
                input = output.0;
                info = output.1;
            }
            Ok((input, info))
        },
    )
}

pub fn on_mark<I: 'static>(
    f: Box<Function<I, I>>,
    key: String,
    value: String,
) -> Box<Function<I, I>> {
    Box::new(move |input, info| -> anyhow::Result<(I, TextDataInfo)> {
        match info.marks.get(&key) {
            Some(mark) if mark == &value => f(input, info),
            _ => Ok((input, info)),
        }
    })
}

pub type Chat = Vec<ChatMessage>;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ChatMessage {
    pub text: String,
    pub role: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChatTemplate {
    pub start: Option<String>,
    pub roles: HashMap<String, String>,
    pub end: Option<String>,
}

impl ChatTemplate {
    pub fn format(&self, chat: &[ChatMessage]) -> anyhow::Result<String> {
        let mut text = String::new();
        if let Some(start) = &self.start {
            text.push_str(start);
        }
        for msg in chat {
            let role_template = self
                .roles
                .get(&msg.role)
                .ok_or_else(|| anyhow!("role {} not found in template", &msg.role))?;
            let formatted = role_template.replace("{text}", &msg.text);
            text.push_str(&formatted);
        }
        if let Some(end) = &self.end {
            text.push_str(end);
        }
        Ok(text)
    }
}

impl<'a> FromPyObject<'a> for ChatTemplate {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let d: &PyDict = ob.extract()?;

        let Some(roles) = d.get_item("roles")? else {
            return Err(py_required_key_error("roles", "chat template"));
        };

        Ok(ChatTemplate {
            start: d.get_item("start")?.map(|s| s.extract()).transpose()?,
            roles: roles.extract()?,
            end: d.get_item("end")?.map(|s| s.extract()).transpose()?,
        })
    }
}

#[cfg(test)]
mod test {
    use crate::data::utils::{ChatMessage, ChatTemplate};

    #[test]
    fn test_chat_template() {
        let template = ChatTemplate {
            start: Some("<start>".to_string()),
            roles: vec![("user", "User: {text}\n"), ("bot", "Bot: {text}")]
                .into_iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            end: Some("<end>".to_string()),
        };
        let chat = vec![
            ChatMessage {
                text: "Hello".to_string(),
                role: "user".to_string(),
            },
            ChatMessage {
                text: "Hi".to_string(),
                role: "bot".to_string(),
            },
        ];
        assert_eq!(
            template.format(&chat).unwrap(),
            "<start>User: Hello\nBot: Hi<end>"
        );
    }
}
