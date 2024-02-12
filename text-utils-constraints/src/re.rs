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

pub struct RegularExpressionConstraint {
    dfa: DFA<Vec<u32>>,
    pattern: String,
    continuations: Vec<Vec<u8>>,
}

impl RegularExpressionConstraint {
    pub fn new(pattern: &str, continuations: &[Vec<u8>]) -> Result<Self, Box<dyn Error>> {
        let dfa = DFA::new(&make_anchored(pattern))?;
        Ok(RegularExpressionConstraint {
            dfa,
            pattern: pattern.to_string(),
            continuations: continuations.to_vec(),
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
        Ok(RegularExpressionConstraint {
            dfa,
            pattern,
            continuations: continuations.to_vec(),
        })
    }
}

impl RegularExpressionConstraint {
    #[allow(dead_code)]
    fn pattern(&self) -> &str {
        &self.pattern
    }

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
    fn is_continuation(&self, mut state: StateID, continuation: &[u8]) -> Option<StateID> {
        for &b in continuation {
            state = self.dfa.next_state(state, b);
            if !self.is_maybe_match(state) {
                return None;
            }
        }
        Some(state)
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
    type State = u32;

    fn get_state(&self, prefix: &[u8]) -> Self::State {
        let start = self
            .dfa
            .start_state_forward(&Input::new(prefix))
            .expect("failed to get start state");
        self.drive(start, prefix).as_u32()
    }

    fn is_match_state(&self, state: Self::State) -> bool {
        self.dfa
            .is_match_state(self.dfa.next_eoi_state(StateID::try_from(state).unwrap()))
    }

    fn get_valid_continuations_with_prefix(&self, prefix: &[u8]) -> (Vec<usize>, Vec<Self::State>) {
        let Ok(start) = self.dfa.start_state_forward(&Input::new(prefix)) else {
            return (vec![], vec![]);
        };
        let state = self.drive(start, prefix);
        self.continuations.iter().enumerate().fold(
            (vec![], vec![]),
            |(mut indices, mut states), (i, cont)| {
                if let Some(state) = self.is_continuation(state, cont) {
                    indices.push(i);
                    states.push(state.as_u32());
                }
                (indices, states)
            },
        )
    }

    fn get_valid_continuations_with_state(
        &self,
        state: Self::State,
    ) -> (Vec<usize>, Vec<Self::State>) {
        self.continuations.iter().enumerate().fold(
            (vec![], vec![]),
            |(mut indices, mut states), (i, cont)| {
                if let Some(state) = self.is_continuation(StateID::try_from(state).unwrap(), cont) {
                    indices.push(i);
                    states.push(state.as_u32());
                }
                (indices, states)
            },
        )
    }
}

#[cfg(test)]
mod test {
    use std::{fs, path::PathBuf};

    use rand::seq::IteratorRandom;

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
        let re = RegularExpressionConstraint::new(r"^ab$", &conts).unwrap();
        let (conts, states) = re.get_valid_continuations_with_prefix(b"");
        assert_eq!(conts, vec![0, 3]);
        assert!(!re.is_match_state(states[0]));
        let (conts, states) = re.get_valid_continuations_with_state(states[0]);
        assert_eq!(conts, vec![1]);
        assert!(re
            .get_valid_continuations_with_state(states[0])
            .0
            .is_empty());
        assert!(re.is_match_state(states[0]));
        let (conts, _) = re.get_valid_continuations_with_prefix(b"a");
        assert_eq!(conts, vec![1]);
        let (conts, _) = re.get_valid_continuations_with_prefix(b"c");
        assert!(conts.is_empty());
    }

    #[test]
    fn test_re_patterns() {
        let continuations = load_continuations();
        let mut rng = rand::thread_rng();
        let n = 10;
        for pat in load_patterns() {
            let re = RegularExpressionConstraint::new(&pat, &continuations).unwrap();
            println!(
                "memory usage: {:.2}kB",
                re.dfa.memory_usage() as f32 / 1000.0
            );
            println!("pattern:\n{}", re.pattern());
            for i in 0..n {
                let mut state = re.get_state(b"");
                let mut is_match = false;
                let mut decoded = vec![];
                while !is_match {
                    let (conts, states) = re.get_valid_continuations_with_state(state);
                    // random sample index between 0 and conts.len()
                    let Some(idx) = (0..conts.len()).choose(&mut rng) else {
                        break;
                    };
                    let cont = conts[idx];
                    decoded.extend(continuations[cont].iter().copied());
                    state = states[idx];
                    is_match = re.is_match_state(state);
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
            let re = RegularExpressionConstraint::from_file(path, &continuations).unwrap();
            println!(
                "memory usage: {:.2}kB",
                re.dfa.memory_usage() as f32 / 1000.0
            );
            println!("pattern:\n{}", re.pattern());
            for i in 0..n {
                let mut state = re.get_state(b"");
                let mut is_match = false;
                let mut decoded = vec![];
                while !is_match {
                    let (conts, states) = re.get_valid_continuations_with_state(state);
                    // random sample index between 0 and conts.len()
                    let Some(idx) = (0..conts.len()).choose(&mut rng) else {
                        break;
                    };
                    let cont = conts[idx];
                    decoded.extend(continuations[cont].iter().copied());
                    state = states[idx];
                    is_match = re.is_match_state(state);
                }
                assert!(is_match);
                println!("{}. sample:\n{}", i + 1, String::from_utf8_lossy(&decoded));
            }
        }
    }
}
