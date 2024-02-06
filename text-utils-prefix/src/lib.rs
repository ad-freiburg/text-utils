use itertools::Itertools;
use rayon::prelude::*;

pub mod art;
pub mod patricia;
pub mod vec;

pub use art::AdaptiveRadixTrie;
pub use patricia::PatriciaTrie;
pub use vec::{PrefixVec, PrefixVecContinuations};

pub trait PrefixSearch {
    type Value;

    fn insert(&mut self, key: &[u8], value: Self::Value) -> Option<Self::Value>;

    fn delete(&mut self, key: &[u8]) -> Option<Self::Value>;

    fn get(&self, key: &[u8]) -> Option<&Self::Value>;

    fn contains_prefix(&self, prefix: &[u8]) -> bool;

    fn path<'a>(&'a self, prefix: &[u8]) -> Vec<(usize, &'a Self::Value)>
    where
        Self::Value: 'a;
}

pub trait ContinuationSearch: PrefixSearch {
    fn continuations(
        &self,
        prefix: &[u8],
    ) -> Box<dyn Iterator<Item = (Vec<u8>, &Self::Value)> + '_>;

    fn contains_continuation(&self, prefix: &[u8], continuation: &[u8]) -> bool;

    fn contains_continuations(&self, prefix: &[u8], continuations: &[Vec<u8>]) -> Vec<usize> {
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

    fn contains_continuations_optimized(
        &self,
        prefix: &[u8],
        continuations: &[Vec<u8>],
        permutation: &[usize],
        skips: &[usize],
    ) -> Vec<usize> {
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

    fn batch_contains_continuations(
        &self,
        prefixes: &[Vec<u8>],
        continuations: &[Vec<u8>],
    ) -> Vec<Vec<usize>> {
        prefixes
            .iter()
            .map(|p| self.contains_continuations(p, continuations))
            .collect()
    }

    fn batch_contains_continuations_optimized(
        &self,
        prefixes: &[Vec<u8>],
        continuations: &[Vec<u8>],
        permutation: &[usize],
        skips: &[usize],
    ) -> Vec<Vec<usize>> {
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
        let (permutation, skips) = optimized_prefix_order(continuations);
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
        self.trie.contains_continuations_optimized(
            prefix.as_ref(),
            continuations,
            permutation,
            skips,
        )
    }

    pub fn batch_continuation_indices(&self, prefixes: &[Vec<u8>]) -> Vec<Vec<usize>> {
        let (continuations, permutation, skips) = &self.continuations;
        self.trie.batch_contains_continuations_optimized_parallel(
            prefixes,
            continuations,
            permutation,
            skips,
        )
    }
}

#[cfg(test)]
mod test {
    use std::{fs, path::PathBuf};

    use itertools::Itertools;

    use crate::{
        optimized_prefix_order, AdaptiveRadixTrie, ContinuationSearch, PatriciaTrie, PrefixSearch,
        PrefixVec, PrefixVecContinuations,
    };

    fn get_tries() -> Vec<(&'static str, Box<dyn PrefixSearch<Value = usize>>)> {
        vec![
            ("art", Box::new(AdaptiveRadixTrie::default())),
            ("patricia", Box::new(PatriciaTrie::default())),
            ("vec", Box::new(PrefixVec::default())),
        ]
    }

    fn load_prefixes() -> Vec<&'static [u8]> {
        [b"Albert".as_slice(), b"Ber", b"Frank"]
            .into_iter()
            .collect()
    }

    fn load_words() -> Vec<String> {
        let dir = env!("CARGO_MANIFEST_DIR");
        let index = fs::read_to_string(PathBuf::from(dir).join("resources/test/index.100k.txt"))
            .expect("failed to read file");
        index.lines().map(|s| s.to_string()).sorted().collect()
    }

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

    #[test]
    fn test_optimized_prefix_order() {
        let items = ["de", "a", "d", "ab", "abc", "b"];
        let (permutation, skips) = optimized_prefix_order(&items);
        assert_eq!(permutation, vec![1, 3, 4, 5, 2, 0]);
        assert_eq!(skips, vec![2, 1, 0, 0, 1, 0]);
    }

    #[test]
    fn test_prefix_search() {
        for (_, mut trie) in get_tries() {
            assert_eq!(trie.get(b"hello"), None);
            assert_eq!(trie.get(b""), None);
            assert!(!trie.contains_prefix(b""));
            trie.insert(b"", 4);
            trie.insert(b"h", 5);
            trie.insert(b"hello", 1);
            assert_eq!(trie.delete(b"hello"), Some(1));
            assert_eq!(trie.delete(b"hello "), None);
            trie.insert(b"hello", 1);
            trie.insert(b"hell", 2);
            trie.insert(b"hello world", 3);
            assert_eq!(trie.path(b""), vec![(0, &4)]);
            assert_eq!(
                trie.path(b"hello"),
                vec![(0, &4), (1, &5), (4, &2), (5, &1)]
            );
            assert_eq!(trie.get(b"hello"), Some(&1));
            assert_eq!(trie.get(b"hell"), Some(&2));
            assert_eq!(trie.get(b"hello world"), Some(&3));
            assert_eq!(trie.contains_prefix(b"hell"), true);
            assert_eq!(trie.contains_prefix(b"hello"), true);
            assert_eq!(trie.contains_prefix(b""), true);
            assert_eq!(trie.contains_prefix(b"hello world!"), false);
            assert_eq!(trie.contains_prefix(b"test"), false);
            assert_eq!(trie.get(b"hello"), Some(&1));
            assert_eq!(trie.delete(b"hello"), Some(1));
            assert_eq!(trie.get(b"hello"), None);
        }
    }

    #[test]
    fn test_path() {
        let words = load_words();
        let prefixes = load_prefixes();

        for (_, mut trie) in get_tries() {
            words.iter().enumerate().for_each(|(i, w)| {
                trie.insert(w.as_bytes(), i);
            });

            for prefix in &prefixes {
                let path = trie.path(prefix);
                assert!(path
                    .iter()
                    .all(|&(n, i)| { &prefix[..n] == words[*i].as_bytes() }));
                assert!(words.iter().enumerate().all(|(i, w)| {
                    if prefix.starts_with(w.as_bytes()) {
                        path.iter().any(|(_, &idx)| idx == i)
                    } else {
                        path.iter().all(|(_, &idx)| idx != i)
                    }
                }));
            }
        }
    }

    #[test]
    fn test_insert_delete_contains_prefix() {
        let words = load_words();

        for (_, mut trie) in get_tries() {
            words.iter().enumerate().for_each(|(i, w)| {
                trie.insert(w.as_bytes(), i);
            });

            for (i, word) in words.iter().enumerate() {
                assert_eq!(trie.get(word.as_bytes()), Some(&i));
                let bytes = word.as_bytes();
                assert!(trie.contains_prefix(&bytes[..=bytes.len() / 2]));
            }

            for (i, word) in words.iter().enumerate() {
                let even = i % 2 == 0;
                if even {
                    assert_eq!(trie.delete(word.as_bytes()), Some(i));
                    assert_eq!(trie.get(word.as_bytes()), None);
                } else {
                    assert_eq!(trie.get(word.as_bytes()), Some(&i));
                }
            }
        }
    }

    #[test]
    fn test_continuation_vec() {
        let words = load_words();
        let prefixes = load_prefixes();
        let continuations = load_continuations().into_iter().skip(4).collect::<Vec<_>>();

        let vec = PrefixVecContinuations::new(
            words.iter().enumerate().map(|(i, w)| (w, i)).collect(),
            &continuations,
        );

        for prefix in prefixes {
            let conts: Vec<_> = vec.continuations(prefix).map(|(w, v)| (w, *v)).collect();
            // check that no other words than the given conts start with the prefix
            assert!(words.iter().all(|w| {
                let w = w.as_bytes();
                if w.starts_with(prefix) {
                    conts.iter().any(|(c, _)| w == c)
                } else {
                    conts.iter().all(|(c, _)| w != c)
                }
            }));
            for (word, idx) in &conts {
                assert!(word.starts_with(prefix));
                assert_eq!(vec.vec.get(word), Some(idx));
                assert_eq!(words[*idx].as_bytes(), word);
            }
            let cont_indices = vec.contains_continuations(prefix);
            for (i, cont) in continuations.iter().enumerate() {
                let full_prefix: Vec<_> = prefix.iter().chain(cont.iter()).copied().collect();
                let in_conts = conts.iter().any(|(w, _)| w.starts_with(&full_prefix));
                assert!(
                    if cont_indices.contains(&i) {
                        in_conts
                    } else {
                        !in_conts
                    },
                    "prefix: '{}', full prefix: '{}', cont: {:?} ({}, {in_conts}, {i}, '{}')",
                    String::from_utf8_lossy(prefix),
                    String::from_utf8_lossy(&full_prefix),
                    &continuations[i],
                    cont_indices.contains(&i),
                    String::from_utf8_lossy(&continuations[i])
                );
            }
        }
    }

    #[test]
    fn test_continuation_search() {
        let words = load_words();
        let prefixes = load_prefixes();
        let continuations = load_continuations();

        let tries: Vec<(_, Box<dyn ContinuationSearch<Value = usize>>)> = vec![
            ("art", Box::new(AdaptiveRadixTrie::default())),
            ("patricia", Box::new(AdaptiveRadixTrie::default())),
        ];

        for (_, mut trie) in tries {
            words.iter().enumerate().for_each(|(i, w)| {
                trie.insert(w.as_bytes(), i);
            });

            for prefix in &prefixes {
                let conts: Vec<_> = trie.continuations(prefix).map(|(w, v)| (w, *v)).collect();
                // check that no other words than the given conts start with the prefix
                assert!(words.iter().all(|w| {
                    let w = w.as_bytes();
                    if w.starts_with(prefix) {
                        conts.iter().any(|(c, _)| w == c)
                    } else {
                        conts.iter().all(|(c, _)| w != c)
                    }
                }));
                for (word, idx) in &conts {
                    assert!(word.starts_with(prefix));
                    assert_eq!(trie.get(word), Some(idx));
                    assert_eq!(words[*idx].as_bytes(), word);
                }
                let cont_indices = trie.contains_continuations(prefix, &continuations);
                for (i, cont) in continuations.iter().enumerate() {
                    let full_prefix: Vec<_> = prefix.iter().chain(cont.iter()).copied().collect();
                    let contains_cont = trie.contains_continuation(prefix, cont);
                    let in_conts = conts.iter().any(|(w, _)| w.starts_with(&full_prefix));
                    let all = contains_cont && in_conts;
                    assert!(if cont_indices.contains(&i) { all } else { !all });
                }
            }
        }
    }
}
