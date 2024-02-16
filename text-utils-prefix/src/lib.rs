use rayon::prelude::*;

pub mod art;
pub mod patricia;
pub mod utils;
pub mod vec;

pub use art::AdaptiveRadixTrie;
pub use patricia::PatriciaTrie;
pub use vec::{ContinuationsVec, PrefixVec};

pub trait PrefixSearch {
    type Value;

    fn insert(&mut self, key: &[u8], value: Self::Value) -> Option<Self::Value>;

    fn delete(&mut self, key: &[u8]) -> Option<Self::Value>;

    fn get(&self, key: &[u8]) -> Option<&Self::Value>;

    fn contains_prefix(&self, prefix: &[u8]) -> bool;

    fn path(&self, prefix: &[u8]) -> Vec<(usize, &Self::Value)>;

    fn continuations(
        &self,
        prefix: &[u8],
    ) -> Box<dyn Iterator<Item = (Vec<u8>, &Self::Value)> + '_>;
}

pub trait ContinuationSearch {
    fn contains_continuations(&self, prefix: &[u8]) -> Vec<usize>;

    #[inline]
    fn batch_contains_continuations(&self, prefixes: &[Vec<u8>]) -> Vec<Vec<usize>> {
        prefixes
            .iter()
            .map(|prefix| self.contains_continuations(prefix))
            .collect()
    }

    #[inline]
    fn batch_contains_continuations_parallel(&self, prefixes: &[Vec<u8>]) -> Vec<Vec<usize>>
    where
        Self: Sync,
    {
        prefixes
            .par_iter()
            .map(|prefix| self.contains_continuations(prefix))
            .collect()
    }
}

pub trait ContinuationsTrie {
    fn contains_continuations(
        &self,
        prefix: &[u8],
        continuations: &[Vec<u8>],
        permutation: &[usize],
        skips: &[usize],
    ) -> Vec<usize>;
}

pub struct ContinuationTrie<T> {
    pub trie: T,
    continuations: Vec<Vec<u8>>,
    optimized: (Vec<usize>, Vec<usize>),
}

impl<T> ContinuationTrie<T> {
    pub fn new(trie: T, continuations: Vec<Vec<u8>>) -> Self {
        let optimized = utils::optimized_prefix_order(&continuations);
        Self {
            trie,
            continuations,
            optimized,
        }
    }
}

impl<T> ContinuationSearch for ContinuationTrie<T>
where
    T: ContinuationsTrie,
{
    fn contains_continuations(&self, prefix: &[u8]) -> Vec<usize> {
        let (permutation, skips) = &self.optimized;
        self.trie
            .contains_continuations(prefix, &self.continuations, permutation, skips)
    }
}

impl<T> PrefixSearch for ContinuationTrie<T>
where
    T: PrefixSearch,
{
    type Value = T::Value;

    fn insert(&mut self, key: &[u8], value: Self::Value) -> Option<Self::Value> {
        self.trie.insert(key, value)
    }

    fn delete(&mut self, key: &[u8]) -> Option<Self::Value> {
        self.trie.delete(key)
    }

    fn get(&self, key: &[u8]) -> Option<&Self::Value> {
        self.trie.get(key)
    }

    fn contains_prefix(&self, prefix: &[u8]) -> bool {
        self.trie.contains_prefix(prefix)
    }

    fn path(&self, prefix: &[u8]) -> Vec<(usize, &Self::Value)> {
        self.trie.path(prefix)
    }

    fn continuations(
        &self,
        prefix: &[u8],
    ) -> Box<dyn Iterator<Item = (Vec<u8>, &Self::Value)> + '_> {
        self.trie.continuations(prefix)
    }
}

#[cfg(test)]
mod test {
    use std::{fs, path::PathBuf};

    use itertools::Itertools;
    use rand::{seq::SliceRandom, Rng};

    use crate::{
        AdaptiveRadixTrie, ContinuationSearch, ContinuationTrie, ContinuationsTrie,
        ContinuationsVec, PatriciaTrie, PrefixSearch, PrefixVec,
    };

    fn get_prefix_searchers() -> Vec<(&'static str, Box<dyn PrefixSearch<Value = usize>>)> {
        vec![
            ("art", Box::new(AdaptiveRadixTrie::default())),
            ("patricia", Box::new(PatriciaTrie::default())),
            ("vec", Box::new(PrefixVec::default())),
        ]
    }

    fn load_prefixes(words: &[String], n: usize) -> Vec<&[u8]> {
        // sample n random prefixes from the words
        let mut rng = rand::thread_rng();
        words
            .choose_multiple(&mut rng, n)
            .into_iter()
            .map(|s| {
                let s = s.as_bytes();
                // choose random prefix length
                let len = rng.gen_range(0..s.len());
                &s[..len.max(2).min(s.len())]
            })
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
    fn test_prefix_search() {
        for (_, mut pfx) in get_prefix_searchers() {
            assert_eq!(pfx.get(b"hello"), None);
            assert_eq!(pfx.get(b""), None);
            assert!(!pfx.contains_prefix(b""));
            pfx.insert(b"", 4);
            pfx.insert(b"h", 5);
            pfx.insert(b"hello", 1);
            assert_eq!(pfx.delete(b"hello"), Some(1));
            assert_eq!(pfx.delete(b"hello "), None);
            pfx.insert(b"hello", 1);
            pfx.insert(b"hell", 2);
            pfx.insert(b"hello world", 3);
            assert_eq!(pfx.path(b""), vec![(0, &4)]);
            assert_eq!(pfx.path(b"hello"), vec![(0, &4), (1, &5), (4, &2), (5, &1)]);
            assert_eq!(pfx.get(b"hello"), Some(&1));
            assert_eq!(pfx.get(b"hell"), Some(&2));
            assert_eq!(pfx.get(b"hello world"), Some(&3));
            assert_eq!(pfx.contains_prefix(b"hell"), true);
            assert_eq!(pfx.contains_prefix(b"hello"), true);
            assert_eq!(pfx.contains_prefix(b""), true);
            assert_eq!(pfx.contains_prefix(b"hello world!"), false);
            assert_eq!(pfx.contains_prefix(b"test"), false);
            assert_eq!(pfx.get(b"hello"), Some(&1));
            assert_eq!(pfx.delete(b"hello"), Some(1));
            assert_eq!(pfx.get(b"hello"), None);
        }
    }

    #[test]
    fn test_path() {
        let words = load_words();
        let prefixes = load_prefixes(&words, 1000);

        for (_, mut pfx) in get_prefix_searchers() {
            words.iter().enumerate().for_each(|(i, w)| {
                pfx.insert(w.as_bytes(), i);
            });

            for prefix in &prefixes {
                let path = pfx.path(prefix);
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

        for (_, mut pfx) in get_prefix_searchers() {
            words.iter().enumerate().for_each(|(i, w)| {
                pfx.insert(w.as_bytes(), i);
            });

            for (i, word) in words.iter().enumerate() {
                assert_eq!(pfx.get(word.as_bytes()), Some(&i));
                let bytes = word.as_bytes();
                assert!(pfx.contains_prefix(&bytes[..=bytes.len() / 2]));
            }

            for (i, word) in words.iter().enumerate() {
                let even = i % 2 == 0;
                if even {
                    assert_eq!(pfx.delete(word.as_bytes()), Some(i));
                    assert_eq!(pfx.get(word.as_bytes()), None);
                } else {
                    assert_eq!(pfx.get(word.as_bytes()), Some(&i));
                }
            }
        }
    }

    #[test]
    fn test_continuation_vec() {
        let words = load_words();
        let prefixes = load_prefixes(&words, 10);
        let continuations = load_continuations();

        let vec = ContinuationsVec::new(
            words.iter().enumerate().map(|(i, w)| (w, i)).collect(),
            &continuations,
        );

        for prefix in prefixes {
            let conts: Vec<_> = vec
                .vec
                .continuations(prefix)
                .map(|(w, v)| (w, *v))
                .collect();
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

    trait PrefixContTrie: PrefixSearch + ContinuationsTrie {}
    impl<T> PrefixContTrie for T where T: PrefixSearch + ContinuationsTrie + ?Sized {}
    impl<T: PrefixSearch + ?Sized> PrefixSearch for Box<T> {
        type Value = T::Value;

        fn insert(&mut self, key: &[u8], value: Self::Value) -> Option<Self::Value> {
            self.as_mut().insert(key, value)
        }

        fn delete(&mut self, key: &[u8]) -> Option<Self::Value> {
            self.as_mut().delete(key)
        }

        fn get(&self, key: &[u8]) -> Option<&Self::Value> {
            self.as_ref().get(key)
        }

        fn contains_prefix(&self, prefix: &[u8]) -> bool {
            self.as_ref().contains_prefix(prefix)
        }

        fn path(&self, prefix: &[u8]) -> Vec<(usize, &Self::Value)> {
            self.as_ref().path(prefix)
        }

        fn continuations(
            &self,
            prefix: &[u8],
        ) -> Box<dyn Iterator<Item = (Vec<u8>, &Self::Value)> + '_> {
            self.as_ref().continuations(prefix)
        }
    }
    impl<T: ContinuationsTrie + ?Sized> ContinuationsTrie for Box<T> {
        fn contains_continuations(
            &self,
            prefix: &[u8],
            continuations: &[Vec<u8>],
            permutation: &[usize],
            skips: &[usize],
        ) -> Vec<usize> {
            self.as_ref()
                .contains_continuations(prefix, continuations, permutation, skips)
        }
    }

    #[test]
    fn test_continuation_tries() {
        let words = load_words();
        let prefixes = load_prefixes(&words, 10);
        let continuations = load_continuations();

        let tries: Vec<(_, ContinuationTrie<Box<dyn PrefixContTrie<Value = usize>>>)> = vec![
            (
                "art",
                ContinuationTrie::new(
                    Box::new(
                        words
                            .iter()
                            .zip(0..words.len())
                            .collect::<AdaptiveRadixTrie<_>>(),
                    ),
                    continuations.clone(),
                ),
            ),
            (
                "patricia",
                ContinuationTrie::new(
                    Box::new(
                        words
                            .iter()
                            .zip(0..words.len())
                            .collect::<PatriciaTrie<_>>(),
                    ),
                    continuations.clone(),
                ),
            ),
        ];

        for (_, trie) in tries {
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
                let cont_indices = trie.contains_continuations(prefix);
                for (i, cont) in continuations.iter().enumerate() {
                    let full_prefix: Vec<_> = prefix.iter().chain(cont.iter()).copied().collect();
                    let in_conts = conts.iter().any(|(w, _)| w.starts_with(&full_prefix));
                    assert!(if cont_indices.contains(&i) {
                        in_conts
                    } else {
                        !in_conts
                    });
                }
            }
        }
    }

    #[test]
    fn test_continuation_search() {
        let words = load_words();
        let prefixes = load_prefixes(&words, 100);
        let continuations = load_continuations();

        let cont_search: Vec<(_, Box<dyn ContinuationSearch>)> = vec![
            (
                "art",
                Box::new(ContinuationTrie::new(
                    words
                        .iter()
                        .zip(0..words.len())
                        .collect::<AdaptiveRadixTrie<_>>(),
                    continuations.clone(),
                )),
            ),
            (
                "patricia",
                Box::new(ContinuationTrie::new(
                    words
                        .iter()
                        .zip(0..words.len())
                        .collect::<PatriciaTrie<_>>(),
                    continuations.clone(),
                )),
            ),
            (
                "vec",
                Box::new(ContinuationsVec::new(
                    words.iter().zip(0..words.len()).collect(),
                    &continuations,
                )),
            ),
        ];

        for prefix in &prefixes {
            let conts: Vec<_> = cont_search
                .iter()
                .map(|(_, c)| {
                    let mut conts = c.contains_continuations(prefix);
                    conts.sort();
                    conts
                })
                .collect();
            assert!(conts.windows(2).all(|w| w[0] == w[1]),);
            assert!(conts[0].iter().all(|&i| {
                let extended_prefix: Vec<_> =
                    prefix.iter().chain(&continuations[i]).copied().collect();
                words
                    .iter()
                    .any(|w| w.as_bytes().starts_with(&extended_prefix))
            }));
        }
    }
}
