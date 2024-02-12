use rayon::prelude::*;

pub mod re;
pub mod utils;

pub use re::RegularExpressionConstraint;

pub trait Constraint {
    type State;

    fn get_state(&self, prefix: &[u8]) -> Self::State;

    fn is_match_state(&self, state: Self::State) -> bool;

    fn get_valid_continuations_with_state(
        &self,
        state: Self::State,
    ) -> (Vec<usize>, Vec<Self::State>);

    fn get_valid_continuations_with_prefix(&self, prefix: &[u8]) -> (Vec<usize>, Vec<Self::State>);

    fn get_valid_continuations_with_states(
        &self,
        states: Vec<Self::State>,
    ) -> (Vec<Vec<usize>>, Vec<Vec<Self::State>>)
    where
        Self: Sync,
        Self::State: Send + Sync,
    {
        states
            .into_par_iter()
            .map(|state| self.get_valid_continuations_with_state(state))
            .collect()
    }

    fn get_valid_continuations_with_prefixes(
        &self,
        prefixes: &[Vec<u8>],
    ) -> (Vec<Vec<usize>>, Vec<Vec<Self::State>>)
    where
        Self: Sync,
        Self::State: Send,
    {
        prefixes
            .par_iter()
            .map(|prefix| self.get_valid_continuations_with_prefix(prefix))
            .collect()
    }
}
