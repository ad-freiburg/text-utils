use itertools::Itertools;
use rayon::prelude::*;

pub mod adaptive_radix_trie;
pub mod patricia_trie;
pub mod trie;

pub trait PrefixSearch {
    type Value;

    fn insert<K>(&mut self, key: K, value: Self::Value)
    where
        K: AsRef<[u8]>;

    fn delete<K>(&mut self, key: K) -> Option<Self::Value>
    where
        K: AsRef<[u8]>;

    fn get<K>(&self, key: K) -> Option<&Self::Value>
    where
        K: AsRef<[u8]>;

    fn contains_prefix<P>(&self, prefix: P) -> bool
    where
        P: AsRef<[u8]>;
}

pub trait ContinuationSearch: PrefixSearch {
    fn continuations<'a, P>(
        &'a self,
        prefix: P,
    ) -> impl Iterator<Item = (Vec<u8>, &'a Self::Value)>
    where
        P: AsRef<[u8]>,
        Self::Value: 'a;

    fn contains_continuation<P, C>(&self, prefix: P, continuation: C) -> bool
    where
        P: AsRef<[u8]>,
        C: AsRef<[u8]>;

    fn contains_continuations<P, C>(&self, prefix: P, continuations: &[C]) -> Vec<usize>
    where
        P: AsRef<[u8]>,
        C: AsRef<[u8]>,
    {
        // default naive implementation, should be overridden if there is a more efficient way
        continuations
            .iter()
            .enumerate()
            .filter_map(|(i, c)| {
                if self.contains_continuation(prefix.as_ref(), c.as_ref()) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }

    fn contains_continuations_optimized<P, C>(
        &self,
        prefix: P,
        continuations: &[C],
        permutation: &[usize],
        skips: &[usize],
    ) -> Vec<usize>
    where
        P: AsRef<[u8]>,
        C: AsRef<[u8]>,
    {
        // default naive implementation, should be overridden if there is a more efficient way
        assert_eq!(continuations.len(), permutation.len());
        assert_eq!(continuations.len(), skips.len());
        let mut result = vec![];
        let mut i = 0;
        while let Some(&j) = permutation.get(i) {
            let continuation = continuations[j].as_ref();
            if self.contains_continuation(prefix.as_ref(), continuation) {
                result.push(j);
            } else {
                i += skips[i];
            };
            i += 1;
        }
        result
    }

    fn batch_contains_continuations<P, C>(
        &self,
        prefixes: &[P],
        continuations: &[C],
    ) -> Vec<Vec<usize>>
    where
        P: AsRef<[u8]>,
        C: AsRef<[u8]>,
    {
        prefixes
            .iter()
            .map(|p| self.contains_continuations(p, continuations))
            .collect()
    }

    fn batch_contains_continuations_optimized<P, C>(
        &self,
        prefixes: &[P],
        continuations: &[C],
        permutation: &[usize],
        skips: &[usize],
    ) -> Vec<Vec<usize>>
    where
        P: AsRef<[u8]>,
        C: AsRef<[u8]>,
    {
        prefixes
            .iter()
            .map(|p| self.contains_continuations_optimized(p, continuations, permutation, skips))
            .collect()
    }

    fn batch_contains_continuations_optimized_parallel(
        &self,
        prefixes: &[Vec<u8>],
        continuations: &[Vec<u8>],
        permutation: &[usize],
        skips: &[usize],
    ) -> Vec<Vec<usize>>
    where
        Self: Sync,
    {
        prefixes
            .par_iter()
            .map(|p| self.contains_continuations_optimized(p, continuations, permutation, skips))
            .collect()
    }
}

pub fn optimized_continuations<C>(continuations: &[C]) -> (Vec<usize>, Vec<usize>)
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
        // if the current continuation is a prefix of the next one, we can skip the
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

pub struct ContinuationTrie<T> {
    trie: T,
    continuations: (Vec<Vec<u8>>, Vec<usize>, Vec<usize>),
}

impl<T> ContinuationTrie<T>
where
    T: ContinuationSearch + Sync,
{
    pub fn new<C>(trie: T, continuations: &[C]) -> Self
    where
        C: AsRef<[u8]>,
    {
        let (permutation, skips) = optimized_continuations(continuations);
        Self {
            trie,
            continuations: (
                continuations.iter().map(|c| c.as_ref().to_vec()).collect(),
                permutation,
                skips,
            ),
        }
    }

    pub fn continuation_indices<P>(&self, prefix: P) -> Vec<usize>
    where
        P: AsRef<[u8]>,
    {
        let (continuations, permutation, skips) = &self.continuations;
        self.trie
            .contains_continuations_optimized(prefix, continuations, permutation, skips)
    }

    pub fn batch_continuation_indices(&self, prefixes: &[Vec<u8>]) -> Vec<Vec<usize>> {
        let (continuations, permutation, skips) = &self.continuations;
        self.trie.batch_contains_continuations_optimized(
            prefixes,
            continuations,
            permutation,
            skips,
        )
    }
}

#[cfg(test)]
mod test {
    use crate::optimized_continuations;

    #[test]
    fn test_optimized_continuations() {
        let continuations = ["de", "a", "d", "ab", "abc", "b"];
        let (permutation, skips) = optimized_continuations(&continuations);
        assert_eq!(permutation, vec![1, 3, 4, 5, 2, 0]);
        assert_eq!(skips, vec![2, 1, 0, 0, 1, 0]);
    }
}
