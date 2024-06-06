use rayon::prelude::*;
use wasm_bindgen::prelude::*;

pub mod lr1;
pub mod re;
pub mod utils;

pub use re::RegularExpressionConstraint;
pub use regex_automata::util::primitives::StateID as RegularExpressionState;

pub use lr1::{
    ExactLR1GrammarConstraint, LR1GrammarConstraint, LR1GrammarParser, LR1NextState, LR1Parse,
    LR1State,
};

pub trait Constraint {
    type State;
    type NextState;

    fn get_state(&self, prefix: &[u8]) -> Option<Self::State>;

    fn get_start_state(&self) -> Self::State;

    fn is_match_state(&self, state: &Self::State) -> bool;

    fn get_valid_continuations_with_state(
        &self,
        state: &Self::State,
    ) -> (Vec<usize>, Vec<Self::NextState>);

    fn get_valid_continuations_with_states(
        &self,
        states: &[Self::State],
    ) -> (Vec<Vec<usize>>, Vec<Vec<Self::NextState>>)
    where
        Self: Sync,
        Self::State: Send + Sync,
        Self::NextState: Send + Sync,
    {
        states
            .into_par_iter()
            .map(|state| self.get_valid_continuations_with_state(state))
            .collect()
    }
}

#[wasm_bindgen]
pub fn parse(text: &str, grammar: &str, lexer: &str) -> Option<String> {
    let parser = LR1GrammarParser::new(grammar, lexer).ok()?;
    let parse = parser.parse(text, true, true).ok()?;
    Some(parse.pretty(text, true, true))
}
