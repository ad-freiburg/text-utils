use std::{
    error::Error,
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use crate::Constraint;
use regex_automata::{
    dfa::{dense::DFA, Automaton},
    util::primitives::StateID,
    Input,
};

pub fn make_anchored(pat: &str) -> String {
    match (pat.starts_with('^'), pat.ends_with('$')) {
        (true, true) => pat.to_string(),
        (true, false) => format!("^({})$", pat.chars().skip(1).collect::<String>()),
        (false, true) => {
            let mut chars = pat.chars().collect::<Vec<_>>();
            chars.pop();
            format!("^({})$", chars.into_iter().collect::<String>())
        }
        (false, false) => format!("^({})$", pat),
    }
}

#[derive(Clone)]
pub struct RegularExpressionConstraint {
    dfa: DFA<Vec<u32>>,
    pattern: String,
    continuations: Vec<Vec<u8>>,
    state: StateID,
}

impl RegularExpressionConstraint {
    pub fn new(pattern: &str, continuations: &[Vec<u8>]) -> Result<Self, Box<dyn Error>> {
        let dfa = DFA::new(&make_anchored(pattern))?;
        let state = dfa.start_state_forward(&Input::new(""))?;
        Ok(RegularExpressionConstraint {
            dfa,
            pattern: pattern.to_string(),
            continuations: continuations.to_vec(),
            state,
        })
    }

    pub fn from_file(
        path: impl AsRef<Path>,
        continuations: &[Vec<u8>],
    ) -> Result<Self, Box<dyn Error>> {
        let reader = BufReader::new(File::open(path.as_ref())?);
        let mut pattern = String::new();
        for line in reader.lines() {
            let line = line?;
            if line.starts_with('#') {
                continue;
            } else {
                pattern.push_str(&line);
                pattern.push('\n');
            }
        }
        // remove last new line
        pattern.pop();
        let dfa = DFA::new(&make_anchored(&pattern))?;
        let state = dfa.start_state_forward(&Input::new(""))?;
        Ok(RegularExpressionConstraint {
            dfa,
            pattern,
            continuations: continuations.to_vec(),
            state,
        })
    }
}

impl RegularExpressionConstraint {
    #[inline]
    fn is_maybe_match(&self, state: StateID) -> bool {
        if self.dfa.is_dead_state(state) || self.dfa.is_quit_state(state) {
            false
        } else {
            // non-special, match, start or accelerated states can be a match
            true
        }
    }

    #[inline]
    fn is_valid_continuation(&self, mut state: StateID, continuation: &[u8]) -> bool {
        for &b in continuation {
            state = self.dfa.next_state(state, b);
            if !self.is_maybe_match(state) {
                return false;
            }
        }
        true
    }

    #[inline]
    fn drive(&self, state: StateID, continuation: &[u8]) -> StateID {
        let mut state = state;
        for &b in continuation {
            state = self.dfa.next_state(state, b);
        }
        state
    }
}

impl Constraint for RegularExpressionConstraint {
    fn set_prefix(&mut self, prefix: &[u8]) {
        let start = self
            .dfa
            .start_state_forward(&Input::new(prefix))
            .expect("failed to get start state");
        self.state = self.drive(start, prefix);
    }

    fn get_valid_continuations(&self) -> Vec<usize> {
        self.continuations
            .iter()
            .enumerate()
            .filter_map(|(i, cont)| {
                if self.is_valid_continuation(self.state, cont) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }

    fn add_continuation(&mut self, continuation: usize) -> bool {
        self.state = self.drive(self.state, &self.continuations[continuation]);
        self.dfa.is_match_state(self.dfa.next_eoi_state(self.state))
    }

    fn get_valid_continuations_with_prefix(&self, prefix: &[u8]) -> Vec<usize> {
        let Ok(start) = self.dfa.start_state_forward(&Input::new(prefix)) else {
            return vec![];
        };
        let state = self.drive(start, prefix);
        self.continuations
            .iter()
            .enumerate()
            .filter_map(|(i, cont)| {
                if self.is_valid_continuation(state, cont) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod test {
    use std::{fs, path::PathBuf};

    use rand::seq::SliceRandom;

    use super::*;

    fn load_continuations() -> Vec<Vec<u8>> {
        let dir = env!("CARGO_MANIFEST_DIR");
        let continuations_json =
            fs::read(PathBuf::from(dir).join("resources/test/continuations.json"))
                .expect("failed to read file");

        // use serde to deserialize continuations array from json
        serde_json::from_slice::<Vec<String>>(&continuations_json)
            .unwrap()
            .into_iter()
            .map(|c| c.as_bytes().to_vec())
            .collect()
    }

    fn load_patterns() -> Vec<String> {
        ["yes|no", "[0-9]{5}", r"[a-z]{10}@[a-z]{10}\.(com|org|de)"]
            .iter()
            .map(|s| make_anchored(s))
            .collect()
    }

    #[test]
    fn test_make_anchored() {
        assert_eq!(make_anchored("a"), "^(a)$");
        assert_eq!(make_anchored("^a"), "^(a)$");
        assert_eq!(make_anchored("a$"), "^(a)$");
        assert_eq!(make_anchored("^a$"), "^a$");
    }

    #[test]
    fn test_re_simple() {
        let conts: Vec<_> = ["a", "b", "aa", "ab"]
            .iter()
            .map(|s| s.as_bytes().to_vec())
            .collect();
        let mut re = RegularExpressionConstraint::new(r"^ab$", &conts).unwrap();
        let mut conts = re.get_valid_continuations();
        conts.sort();
        assert_eq!(conts, vec![0, 3]);
        let is_match = re.add_continuation(0);
        assert_eq!(re.get_valid_continuations(), vec![1]);
        assert!(!is_match);
        let is_match = re.add_continuation(1);
        assert!(re.get_valid_continuations().is_empty());
        assert!(is_match);
        re.set_prefix(b"a");
        let conts = re.get_valid_continuations();
        assert_eq!(conts, vec![1]);
        re.set_prefix(b"c");
        let conts = re.get_valid_continuations();
        assert!(conts.is_empty());
    }

    #[test]
    fn test_re_patterns() {
        let continuations = load_continuations();
        let mut rng = rand::thread_rng();
        let n = 10;
        for pat in load_patterns() {
            let mut re = RegularExpressionConstraint::new(&pat, &continuations).unwrap();
            println!(
                "memory usage: {:.2}kB",
                re.dfa.memory_usage() as f32 / 1000.0
            );
            println!("pattern:\n{}", re.pattern);
            for i in 0..n {
                re.set_prefix(b"");
                let mut is_match = false;
                let mut decoded = vec![];
                while !is_match {
                    let conts = re.get_valid_continuations();
                    // random sample cont
                    let Some(cont) = conts.choose(&mut rng) else {
                        break;
                    };
                    is_match = re.add_continuation(*cont);
                    decoded.extend(continuations[*cont].iter().copied());
                }
                assert!(is_match);
                println!("{}. sample:\n{}", i + 1, String::from_utf8_lossy(&decoded));
            }
        }
    }

    #[test]
    fn test_re_files() {
        let continuations = load_continuations();
        let files = ["rdf_triples.txt", "template.txt", "json.txt"];
        let mut rng = rand::thread_rng();
        let n = 10;
        for file in files {
            let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("resources/test")
                .join(file);
            let mut re = RegularExpressionConstraint::from_file(path, &continuations).unwrap();
            println!(
                "memory usage: {:.2}kB",
                re.dfa.memory_usage() as f32 / 1000.0
            );
            println!("pattern:\n{}", re.pattern);
            for i in 0..n {
                re.set_prefix(b"");
                let mut is_match = false;
                let mut decoded = vec![];
                while !is_match {
                    let conts = re.get_valid_continuations();
                    // random sample cont
                    let Some(cont) = conts.choose(&mut rng) else {
                        break;
                    };
                    is_match = re.add_continuation(*cont);
                    decoded.extend(continuations[*cont].iter().copied());
                }
                assert!(is_match);
                println!("{}. sample:\n{}", i + 1, String::from_utf8_lossy(&decoded));
            }
        }
    }
}
