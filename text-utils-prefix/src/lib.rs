pub mod art;
pub mod patricia;
pub mod utils;
pub mod vec;

pub use art::{AdaptiveRadixTrie, ArtContinuationTrie, ArtMmapContinuationTrie};
pub use patricia::PatriciaTrie;
pub use utils::optimized_continuation_permutation;
pub use vec::{PrefixContinuationVec, PrefixVec};

pub trait PrefixSearch {
    type Value;

    fn insert(&mut self, key: &[u8], value: Self::Value) -> Option<Self::Value>;

    fn delete(&mut self, key: &[u8]) -> Option<Self::Value>;

    fn get(&self, key: &[u8]) -> Option<&Self::Value>;

    fn contains(&self, prefix: &[u8]) -> bool;

    fn values_along_path(&self, prefix: &[u8]) -> Vec<(usize, &Self::Value)>;

    fn continuations(
        &self,
        prefix: &[u8],
    ) -> Box<dyn Iterator<Item = (Vec<u8>, &Self::Value)> + '_>;
}

#[cfg(test)]
mod test {
    use std::{fs, path::PathBuf};

    use itertools::Itertools;
    use rand::{seq::SliceRandom, Rng};

    use crate::{
        utils, AdaptiveRadixTrie, ArtContinuationTrie, PatriciaTrie, PrefixContinuationVec,
        PrefixSearch, PrefixVec,
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
            .map(|c| {
                c.as_bytes()
                    .into_iter()
                    .filter(|&b| *b > 0)
                    .copied()
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_prefix_search() {
        for (name, mut pfx) in get_prefix_searchers() {
            println!("testing {}", name);
            assert_eq!(pfx.get(b"hello"), None);
            assert_eq!(pfx.get(b""), None);
            assert!(!pfx.contains(b""));
            pfx.insert(b"", 4);
            pfx.insert(b"h", 5);
            pfx.insert(b"hello", 1);
            assert_eq!(pfx.delete(b"hello"), Some(1));
            assert_eq!(pfx.delete(b"hello "), None);
            pfx.insert(b"hello", 1);
            pfx.insert(b"hell", 2);
            pfx.insert(b"hello world", 3);
            assert_eq!(pfx.values_along_path(b""), vec![(0, &4)]);
            assert_eq!(
                pfx.values_along_path(b"hello"),
                vec![(0, &4), (1, &5), (4, &2), (5, &1)]
            );
            assert_eq!(pfx.get(b"hello"), Some(&1));
            assert_eq!(pfx.get(b"hell"), Some(&2));
            assert_eq!(pfx.get(b"hello world"), Some(&3));
            assert_eq!(pfx.contains(b"hell"), true);
            assert_eq!(pfx.contains(b"hello"), true);
            assert_eq!(pfx.contains(b""), true);
            assert_eq!(pfx.contains(b"hello world!"), false);
            assert_eq!(pfx.contains(b"test"), false);
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
                let path = pfx.values_along_path(prefix);
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

        for (name, mut pfx) in get_prefix_searchers() {
            if name == "vec" {
                continue;
            }
            words.iter().enumerate().for_each(|(i, w)| {
                pfx.insert(w.as_bytes(), i);
            });

            for (i, word) in words.iter().enumerate() {
                assert_eq!(pfx.get(word.as_bytes()), Some(&i));
                let bytes = word.as_bytes();
                assert!(pfx.contains(&bytes[..=bytes.len() / 2]));
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
    fn test_prefix_continuation_vec() {
        let words = load_words();
        let prefixes = load_prefixes(&words, 10);
        let continuations = load_continuations();

        let vec = PrefixContinuationVec::new(
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
            let cont_indices = vec.continuation_indices(prefix);
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
    fn test_adaptive_radix_continuation_trie() {
        let words = load_words();
        let prefixes = load_prefixes(&words, 10);
        let continuations = load_continuations();
        let (perm, skip) = utils::optimized_continuation_permutation(&continuations);

        let trie = ArtContinuationTrie::new(
            words
                .iter()
                .map(|s| s.as_bytes().to_vec())
                .zip(0..words.len()),
        );

        for prefix in &prefixes {
            let conts: Vec<_> = trie
                .trie
                .continuations(prefix)
                .map(|(w, v)| (w, v.clone()))
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
                assert_eq!(trie.trie.get(word), Some(idx));
                assert_eq!(words[*idx.as_ref()].as_bytes(), word);
            }
            let cont_indices = trie.continuation_indices(prefix, &continuations, &perm, &skip);
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
