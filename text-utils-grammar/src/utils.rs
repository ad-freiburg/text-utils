use std::{collections::HashMap, error::Error, fmt::Debug};

use indexmap::IndexMap;
use itertools::Itertools;
use regex::{escape, Regex};
use regex_automata::{
    dfa::{dense::DFA, Automaton},
    util::primitives::StateID,
    Input,
};

#[derive(Debug)]
pub(crate) enum Part {
    Literal(String),
    Regex(String),
}

pub(crate) fn extract_parts(pattern: &str) -> Vec<Part> {
    let mut parts = vec![];
    for part in pattern.split_whitespace() {
        if (part.starts_with('\'') && part.ends_with('\''))
            || (part.starts_with('"') && part.ends_with('"'))
        {
            // treat part as literal
            parts.push(Part::Literal(escape(&part[1..part.len() - 1])));
        } else {
            // treat part as regular expression
            parts.push(Part::Regex(part.to_string()));
        }
    }
    parts
}

// define function to recursively build pattern from parts
pub(crate) fn pattern_from_parts(
    name: &str,
    parts: &[Part],
    name_regex: &Regex,
    fragments: &HashMap<&str, Vec<Part>>,
    tokens: &IndexMap<&str, Vec<Part>>,
) -> Result<String, Box<dyn Error>> {
    let mut pattern = String::new();
    for part in parts {
        match part {
            Part::Literal(s) => pattern.push_str(s),
            Part::Regex(s) => {
                // find all tokens or framents in regex
                // and replace them with their pattern
                let mut replaced = String::new();
                let mut last_match = 0;
                for caps in name_regex.captures_iter(s) {
                    let m = caps.get(0).unwrap();
                    replaced.push_str(&s[last_match..m.start()]);
                    // surround token or fragment with parentheses to group it
                    replaced.push_str("(?:");
                    let _name = caps.get(1).unwrap().as_str();
                    if let Some(parts) = tokens.get(_name).or_else(|| fragments.get(_name)) {
                        let replacement =
                            pattern_from_parts(name, parts, name_regex, fragments, tokens)?;
                        replaced.push_str(&replacement);
                    } else {
                        return Err(format!(
                            "token or fragment {_name} within {name} not found in lexer"
                        )
                        .into());
                    }
                    replaced.push(')');
                    last_match = m.end();
                }
                replaced.push_str(&s[last_match..]);
                pattern.push_str(&replaced);
            }
        }
    }
    Ok(pattern)
}

fn make_anchored(pat: &str) -> String {
    assert!(!pat.ends_with('$'), "prefix pattern should not end with $");
    if pat.starts_with('^') {
        pat.to_string()
    } else {
        format!("^(?:{})", pat)
    }
}

pub(crate) struct PrefixDFA {
    dfa: DFA<Vec<u32>>,
}

impl Debug for PrefixDFA {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrefixDFA").finish()
    }
}

#[derive(Debug, PartialEq)]
pub(crate) enum PrefixMatch {
    None,
    Full(StateID),
    UpTo(usize),
}

impl PrefixDFA {
    pub(crate) fn new(pattern: &str) -> Result<Self, Box<dyn Error>> {
        let dfa = DFA::new(&make_anchored(pattern))?;
        Ok(PrefixDFA { dfa })
    }

    #[inline]
    fn is_dead_or_quit(&self, state: StateID) -> bool {
        // dead or quit state is an end state
        self.dfa.is_dead_state(state) || self.dfa.is_quit_state(state)
    }

    #[inline]
    fn has_continuation(&self, state: StateID) -> bool {
        (0..=255).any(|b| {
            let next = self.dfa.next_state(state, b);
            !self.is_dead_or_quit(next) || self.is_eoi_match(next)
        })
    }

    #[inline]
    pub(crate) fn drive(&self, mut state: StateID, continuation: &[u8]) -> Option<StateID> {
        for &b in continuation {
            state = self.dfa.next_state(state, b);
            if self.is_dead_or_quit(state) {
                return None;
            }
        }
        // normally we would only check for eoi match,
        // but for a prefix dfa we also need to check for continuations
        if self.is_eoi_match(state) || self.has_continuation(state) {
            Some(state)
        } else {
            None
        }
    }

    #[inline]
    pub(crate) fn get_start_state(&self) -> StateID {
        self.dfa
            .start_state_forward(&Input::new(b""))
            .expect("failed to get start state")
    }

    #[inline]
    pub(crate) fn is_eoi_match(&self, state: StateID) -> bool {
        self.dfa.is_match_state(self.dfa.next_eoi_state(state))
    }

    #[inline]
    pub(crate) fn get_state(&self, prefix: &[u8]) -> Option<StateID> {
        let start = self.get_start_state();
        self.drive(start, prefix)
    }

    #[inline]
    pub(crate) fn find_prefix_match(&self, mut state: StateID, prefix: &[u8]) -> PrefixMatch {
        let mut last_match = None;
        for (i, &b) in prefix.iter().enumerate() {
            state = self.dfa.next_state(state, b);
            if self.dfa.is_match_state(state) {
                last_match = Some(i);
            } else if self.is_dead_or_quit(state) {
                return last_match.map_or(PrefixMatch::None, PrefixMatch::UpTo);
            }
        }
        if self.is_eoi_match(state) || self.has_continuation(state) {
            PrefixMatch::Full(state)
        } else {
            last_match.map_or(PrefixMatch::None, PrefixMatch::UpTo)
        }
    }
}

pub fn optimized_prefix_order<C>(continuations: &[C]) -> (Vec<usize>, Vec<usize>)
where
    C: AsRef<[u8]>,
{
    let permutation: Vec<_> = continuations
        .iter()
        .enumerate()
        .sorted_by(|(_, a), (_, b)| a.as_ref().cmp(b.as_ref()))
        .map(|(i, _)| i)
        .collect();
    let mut skips = vec![0; continuations.len()];
    for i in 0..permutation.len() {
        // if the current key is a prefix of the next one, we can skip the
        // latter
        let continuation = continuations[permutation[i]].as_ref();
        while let Some(next) = permutation.get(i + skips[i] + 1) {
            let next_continuation = continuations[*next].as_ref();
            if next_continuation.starts_with(continuation) {
                skips[i] += 1;
            } else {
                break;
            }
        }
    }
    (permutation, skips)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_drive() {
        let pdfa = PrefixDFA::new("<kgp>.*?</kgp>").unwrap();
        let state = pdfa.get_state(b"<kgp>test</kgp").unwrap();
        assert!(pdfa.drive(state, b">").is_some());
        assert!(pdfa.drive(state, b"a").is_some());
        assert!(pdfa.drive(state, b"> ").is_none());

        let pdfa = PrefixDFA::new("<kgp>.*</kgp>").unwrap();
        let state = pdfa.get_state(b"<kgp>test</kgp").unwrap();
        assert!(pdfa.drive(state, b">").is_some());
        assert!(pdfa.drive(state, b"a").is_some());
        assert!(pdfa.drive(state, b"> ").is_some());
    }

    #[test]
    fn test_prefix_dfa() {
        // simple
        let pdfa = PrefixDFA::new("ab").unwrap();
        assert!(pdfa.get_state(b"").is_some());
        assert!(pdfa.get_state(b"a").is_some());
        assert!(pdfa.get_state(b"ab").is_some());
        assert!(pdfa.get_state(b"b").is_none());
        assert!(pdfa.get_state(b"ab ").is_none());
        let state = pdfa.get_state(b"ab").unwrap();
        assert!(pdfa.is_eoi_match(state));
        assert!(pdfa.drive(pdfa.get_start_state(), b"ab").is_some());
        assert!(pdfa.drive(pdfa.get_start_state(), b"ab ").is_none());

        // simple
        let pdfa = PrefixDFA::new("ab*").unwrap();
        assert!(pdfa.get_state(b"").is_some());
        assert!(pdfa.get_state(b"a").is_some());
        assert!(pdfa.get_state(b"ab").is_some());
        assert!(pdfa.get_state(b"b").is_none());
        assert!(pdfa.get_state(b"ab ").is_none());
        assert!(pdfa.get_state(b"abb").is_some());
        let state = pdfa.get_state(b"abb").unwrap();
        assert!(pdfa.is_eoi_match(state));

        // greedy
        let pdfa = PrefixDFA::new("<a>.*</a>").unwrap();
        assert!(pdfa.get_state(b"<a>test</a>").is_some());
        assert!(pdfa.get_state(b"<a>test</a>t").is_some());
        assert!(pdfa.get_state(b"<a>test</a>tt").is_some());
        let state = pdfa.get_state(b"<a>test</a>").unwrap();
        assert!(!pdfa.is_dead_or_quit(state));
        assert!(pdfa.is_eoi_match(state));

        // non-greedy
        let pdfa = PrefixDFA::new("<a>.*?</a>").unwrap();
        assert!(pdfa.get_state(b"<a>test</a>").is_some());
        assert!(pdfa.get_state(b"<a>test</a>t").is_none());
        assert!(pdfa.get_state(b"<a>test</a>tt").is_none());
        let state = pdfa.get_state(b"<a>test</a>").unwrap();
        assert!(!pdfa.is_dead_or_quit(state));
        assert!(pdfa.is_eoi_match(state));
    }

    #[test]
    fn test_prefix_match() {
        let pdfa = PrefixDFA::new("abcdef").unwrap();
        let state = pdfa.get_state(b"abc").unwrap();
        assert_eq!(
            pdfa.find_prefix_match(pdfa.get_start_state(), b"abc"),
            PrefixMatch::Full(state)
        );
        assert_eq!(pdfa.find_prefix_match(state, b"abc"), PrefixMatch::None);
        assert_eq!(
            pdfa.find_prefix_match(state, b"def"),
            PrefixMatch::Full(pdfa.get_state(b"abcdef").unwrap())
        );
        assert_eq!(
            pdfa.find_prefix_match(pdfa.get_start_state(), b"abcdefg"),
            PrefixMatch::UpTo(6)
        );
        assert_eq!(
            pdfa.find_prefix_match(pdfa.get_state(b"abcdef").unwrap(), b""),
            PrefixMatch::Full(pdfa.get_state(b"abcdef").unwrap())
        );
        assert_eq!(
            pdfa.find_prefix_match(pdfa.get_state(b"abcdef").unwrap(), b"g"),
            PrefixMatch::UpTo(0)
        );
        assert_eq!(pdfa.find_prefix_match(state, b""), PrefixMatch::Full(state));
        let state = pdfa.get_state(b"abcdef").unwrap();
        assert_eq!(
            pdfa.find_prefix_match(pdfa.get_start_state(), b"abcdef"),
            PrefixMatch::Full(state)
        );
        let pdfa = PrefixDFA::new("abcdef+").unwrap();
        let state = pdfa.get_state(b"abcdefff").unwrap();
        assert_eq!(
            pdfa.find_prefix_match(pdfa.get_start_state(), b"abcdefff"),
            PrefixMatch::Full(state)
        );
    }

    #[test]
    fn test_optimized_prefix_order() {
        let items = ["de", "a", "d", "ab", "abc", "b"];
        let (permutation, skips) = optimized_prefix_order(&items);
        assert_eq!(permutation, vec![1, 3, 4, 5, 2, 0]);
        assert_eq!(skips, vec![2, 1, 0, 0, 1, 0]);
    }

    #[test]
    fn test_make_anchored() {
        assert_eq!(make_anchored("a"), "^(?:a)");
        assert_eq!(make_anchored("^a"), "^a");
    }
}
