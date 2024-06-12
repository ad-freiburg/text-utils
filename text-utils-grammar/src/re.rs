use std::{collections::HashMap, error::Error, fs::File, io::read_to_string, path::Path};

use crate::{
    utils::{extract_parts, pattern_from_parts, Part, PrefixDFA},
    Constraint,
};
use indexmap::IndexMap;
use regex::Regex;
use regex_automata::util::primitives::StateID;

pub struct RegularExpressionConstraint {
    pdfa: PrefixDFA,
    continuations: Vec<Vec<u8>>,
}

impl RegularExpressionConstraint {
    pub fn new(content: &str, continuations: Vec<Vec<u8>>) -> Result<Self, Box<dyn Error>> {
        let fragment_name = Regex::new(r"\{([A-Z][A-Z0-9_]*)\}")?;
        let fragment_line = Regex::new(r"(?m)^([A-Z][A-Z0-9_]*)\s+(.+)$")?;
        let sep = Regex::new("(?m)^%%\\n")?;
        let pattern = if let Some(m) = sep.find(content) {
            // parse fragements
            let mut fragments = HashMap::new();
            for line in content[..m.start()].lines() {
                if line.is_empty() || line.trim_start().starts_with("//") {
                    continue;
                }
                let cap = fragment_line
                    .captures(line)
                    .ok_or(format!("invalid fragment line: {line}"))?;
                let name = cap.get(1).unwrap().as_str();
                let pattern = cap.get(2).unwrap().as_str();
                let parts = extract_parts(pattern);
                if fragments.insert(name, parts).is_some() {
                    return Err(format!("duplicate fragment {name}").into());
                };
            }
            pattern_from_parts(
                "regular expression",
                &[Part::Regex(content[m.end()..].to_string())],
                &fragment_name,
                &fragments,
                &IndexMap::new(),
            )?
        } else {
            content.to_string()
        };
        let pdfa = PrefixDFA::new(&pattern)?;
        Ok(RegularExpressionConstraint {
            pdfa,
            continuations,
        })
    }

    pub fn from_file(
        path: impl AsRef<Path>,
        continuations: Vec<Vec<u8>>,
    ) -> Result<Self, Box<dyn Error>> {
        let file = File::open(path.as_ref())?;
        let content = read_to_string(file)?;
        Self::new(&content, continuations)
    }
}

impl Constraint for RegularExpressionConstraint {
    type State = StateID;

    fn get_state(&self, prefix: &[u8]) -> Option<Self::State> {
        self.pdfa.get_state(prefix)
    }

    fn get_start_state(&self) -> Self::State {
        self.pdfa.get_start_state()
    }

    fn is_match_state(&self, state: &Self::State) -> bool {
        self.pdfa.is_eoi_match(*state)
    }

    fn get_valid_continuations(&self, state: &Self::State) -> Vec<usize> {
        self.continuations
            .iter()
            .enumerate()
            .filter_map(|(i, cont)| {
                if self.pdfa.drive(*state, cont).is_some() {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }

    fn get_next_state(&self, state: &Self::State, continuation: usize) -> Option<Self::State> {
        self.pdfa
            .drive(*state, self.continuations.get(continuation)?)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::seq::IteratorRandom;
    use std::{fs, path::PathBuf};

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
        [
            "yes|no",
            "[0-9]{5}",
            r"[a-z]{10}@[a-z]{10}\.(com|org|de)",
            r"Reasoning:\n([1-9]\. .{0, 64}\n){1,9}\nAnswer: (yes|no)",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    #[test]
    fn test_re_simple() {
        let conts: Vec<_> = ["a", "b", "aa", "ab"]
            .iter()
            .map(|s| s.as_bytes().to_vec())
            .collect();
        let re = RegularExpressionConstraint::new(r"^ab", conts.clone()).unwrap();
        let state = re.get_start_state();
        let conts = re.get_valid_continuations(&state);
        assert_eq!(conts, vec![0, 3]);
        let state = re.get_next_state(&state, 0).unwrap();
        assert!(!re.is_match_state(&state));
        let conts = re.get_valid_continuations(&state);
        assert_eq!(conts, vec![1]);
        let state = re.get_next_state(&state, 1).unwrap();
        assert!(re.get_valid_continuations(&state).is_empty());
        assert!(re.is_match_state(&state));
        let state = re.pdfa.get_state(b"a").unwrap();
        let conts = re.get_valid_continuations(&state);
        assert_eq!(conts, vec![1]);
        assert!(re.pdfa.get_state(b"c").is_none());
    }

    #[test]
    fn test_re_patterns() {
        let continuations = load_continuations();
        let mut rng = rand::thread_rng();
        let n = 10;
        for pat in load_patterns() {
            let re = RegularExpressionConstraint::new(&pat, continuations.clone()).unwrap();
            for i in 0..n {
                let mut state = re.get_start_state();
                let mut is_match = false;
                let mut decoded = vec![];
                while !is_match {
                    let conts = re.get_valid_continuations(&state);
                    // random sample index between 0 and conts.len()
                    let Some(idx) = (0..conts.len()).choose(&mut rng) else {
                        break;
                    };
                    let cont = conts[idx];
                    decoded.extend(continuations[cont].iter().copied());
                    state = re.get_next_state(&state, cont).unwrap();
                    is_match = re.is_match_state(&state);
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
            let re = RegularExpressionConstraint::from_file(path, continuations.clone()).unwrap();
            for i in 0..n {
                println!("{file} {i}");
                let mut state = re.get_start_state();
                let mut is_match = false;
                let mut decoded = vec![];
                while !is_match {
                    let conts = re.get_valid_continuations(&state);
                    // random sample index between 0 and conts.len()
                    let Some(idx) = (0..conts.len()).choose(&mut rng) else {
                        break;
                    };
                    let cont = conts[idx];
                    decoded.extend(continuations[cont].iter().copied());
                    state = re.get_next_state(&state, cont).unwrap();
                    is_match = re.is_match_state(&state);
                }
                assert!(is_match);
                println!("{}. sample:\n{}", i + 1, String::from_utf8_lossy(&decoded));
            }
        }
    }
}
