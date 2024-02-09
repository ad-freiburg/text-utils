use rayon::prelude::*;

pub mod re;
pub mod utils;

pub use re::RegularExpressionConstraint;

pub trait Constraint: Clone + Sync {
    fn set_prefix(&mut self, prefix: &[u8]);

    fn add_continuation(&mut self, continuation: usize) -> bool;

    fn get_valid_continuations(&self) -> Vec<usize>;

    fn get_valid_continuations_with_prefix(&self, prefix: &[u8]) -> Vec<usize>;

    fn get_valid_continuations_with_prefixes(&self, prefixes: &[Vec<u8>]) -> Vec<Vec<usize>>
    where
        Self: Sync,
    {
        prefixes
            .par_iter()
            .map(|prefix| self.get_valid_continuations_with_prefix(prefix))
            .collect()
    }
}
