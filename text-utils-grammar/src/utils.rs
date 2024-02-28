use std::error::Error;

use itertools::Itertools;
use regex_automata::{
    dfa::{dense::DFA, Automaton},
    util::primitives::StateID,
    Input,
};

fn make_anchored(pat: &str) -> String {
    let pat: String = match (pat.starts_with('^'), pat.ends_with('$')) {
        (true, true) => return pat.to_string(),
        (true, false) => pat.chars().skip(1).collect(),
        (false, true) => {
            let mut chars = pat.chars().collect::<Vec<_>>();
            chars.pop();
            chars.into_iter().collect()
        }
        (false, false) => pat.to_string(),
    };
    format!("^(?:{})$", pat)
}

pub(crate) struct PrefixDFA {
    dfa: DFA<Vec<u32>>,
}

pub(crate) enum PrefixMatch {
    None,
    Maybe(StateID),
    UpTo(usize, StateID),
}

impl PrefixDFA {
    pub(crate) fn new(pattern: &str) -> Result<Self, Box<dyn Error>> {
        let dfa = DFA::new(&make_anchored(pattern))?;
        Ok(PrefixDFA { dfa })
    }

    #[inline]
    pub(crate) fn get_state(&self, prefix: &[u8]) -> Option<StateID> {
        let start = self.get_start_state();
        self.drive(start, prefix)
    }

    #[inline]
    pub(crate) fn get_start_state(&self) -> StateID {
        self.dfa
            .start_state_forward(&Input::new(b""))
            .expect("failed to get start state")
    }

    #[inline]
    pub(crate) fn is_maybe_match(&self, state: StateID) -> bool {
        // all except dead and quit states can be a match later
        !(self.dfa.is_dead_state(state) || self.dfa.is_quit_state(state))
    }

    #[inline]
    pub(crate) fn is_match_state(&self, state: StateID) -> bool {
        self.dfa.is_match_state(self.dfa.next_eoi_state(state))
    }

    #[inline]
    pub(crate) fn drive(&self, mut state: StateID, continuation: &[u8]) -> Option<StateID> {
        for &b in continuation {
            state = self.dfa.next_state(state, b);
            if !self.is_maybe_match(state) {
                return None;
            }
        }
        Some(state)
    }

    #[inline]
    pub(crate) fn find_prefix_match(&self, mut state: StateID, prefix: &[u8]) -> PrefixMatch {
        let mut last_match = None;
        for (i, &b) in prefix.iter().enumerate() {
            state = self.dfa.next_state(state, b);
            if self.is_match_state(state) {
                last_match = Some((i + 1, state));
            } else if !self.is_maybe_match(state) {
                return last_match.map_or(PrefixMatch::None, |(i, s)| PrefixMatch::UpTo(i, s));
            }
        }
        PrefixMatch::Maybe(state)
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
    fn test_optimized_prefix_order() {
        let items = ["de", "a", "d", "ab", "abc", "b"];
        let (permutation, skips) = optimized_prefix_order(&items);
        assert_eq!(permutation, vec![1, 3, 4, 5, 2, 0]);
        assert_eq!(skips, vec![2, 1, 0, 0, 1, 0]);
    }

    #[test]
    fn test_make_anchored() {
        assert_eq!(make_anchored("a"), "^(a)$");
        assert_eq!(make_anchored("^a"), "^(a)$");
        assert_eq!(make_anchored("a$"), "^(a)$");
        assert_eq!(make_anchored("^a$"), "^a$");
    }
}
