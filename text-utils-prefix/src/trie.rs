use std::collections::HashMap;

use crate::PrefixSearch;

#[derive(Debug)]
struct Node<V> {
    value: Option<V>,
    children: [Option<Box<Node<V>>>; 256],
}

impl<V> Default for Node<V> {
    fn default() -> Self {
        Self {
            value: None,
            children: std::array::from_fn(|_| None),
        }
    }
}

#[derive(Debug)]
pub struct Trie<V> {
    root: Option<Node<V>>,
}

#[derive(Debug)]
pub struct TrieStats {
    pub depth: usize,
    pub num_nodes: usize,
    pub num_keys: usize,
}

impl<V> Trie<V> {
    pub fn stats(&self) -> TrieStats {
        let Some(root) = &self.root else {
            return TrieStats {
                depth: 0,
                num_nodes: 0,
                num_keys: 0,
            };
        };
        let mut stack = vec![(root, 0)];
        let mut max_depth = 0;
        let mut num_keys = 0;
        let mut num_nodes = 0;
        while let Some((node, depth)) = stack.pop() {
            max_depth = max_depth.max(depth);
            num_keys += node.value.is_some() as usize;
            let stack_size = stack.len();
            stack.extend(node.children().map(|child| (child, depth + 1)));
            num_nodes += stack.len() - stack_size;
        }
        TrieStats {
            depth: max_depth,
            num_nodes,
            num_keys,
        }
    }
}

impl<V> Default for Trie<V> {
    fn default() -> Self {
        Self { root: None }
    }
}

impl<K, V> FromIterator<(K, V)> for Trie<V>
where
    K: AsRef<[u8]>,
{
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let mut trie = Self::default();
        for (k, v) in iter {
            trie.insert(k, v);
        }
        trie
    }
}

impl<V> Node<V> {
    #[inline]
    fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    #[inline]
    fn find(&self, key: &[u8]) -> Option<&Self> {
        let mut node = self;
        for k in key {
            let Some(child) = &node.children[*k as usize] else {
                return None;
            };
            node = child;
        }
        Some(node)
    }

    fn children(&self) -> Box<dyn Iterator<Item = &Self> + '_> {
        Box::new(self.children.iter().filter_map(|child| child.as_deref()))
    }
}

impl<V> PrefixSearch for Trie<V> {
    type Value = V;

    fn insert<K>(&mut self, key: K, value: V)
    where
        K: AsRef<[u8]>,
    {
        let mut node = if let Some(node) = &mut self.root {
            node
        } else {
            self.root = Some(Node::default());
            self.root.as_mut().unwrap()
        };
        for k in key.as_ref() {
            let child = &mut node.children[*k as usize];
            if child.is_none() {
                *child = Some(Default::default());
            }
            node = unsafe { child.as_mut().unwrap_unchecked() };
        }
        node.value = Some(value);
    }

    fn delete<K>(&mut self, key: K) -> Option<V>
    where
        K: AsRef<[u8]>,
    {
        let Some(root) = &mut self.root else {
            return None;
        };

        let key = key.as_ref();
        if key.is_empty() {
            let value = root.value.take();
            self.root = None;
            return value;
        }
        let mut node = root;
        for k in key.iter().take(key.len() - 1) {
            let Some(child) = &mut node.children[*k as usize] else {
                return None;
            };
            node = child;
        }
        let last = *key.last()? as usize;
        let Some(child) = &mut node.children[last] else {
            return None;
        };
        if child.is_leaf() {
            node.children[last].take().and_then(|node| node.value)
        } else {
            child.value.take()
        }
    }

    fn get<K>(&self, key: K) -> Option<&V>
    where
        K: AsRef<[u8]>,
    {
        let Some(root) = &self.root else {
            return None;
        };
        root.find(key.as_ref()).and_then(|node| node.value.as_ref())
    }

    fn contains_prefix<P>(&self, prefix: P) -> bool
    where
        P: AsRef<[u8]>,
    {
        let Some(root) = &self.root else {
            return false;
        };
        root.find(prefix.as_ref()).is_some()
    }
}

#[cfg(test)]
mod test {
    use crate::{trie::Trie, PrefixSearch};
    use std::fs;
    use std::path::PathBuf;

    #[test]
    fn test_trie() {
        let mut trie = Trie::default();
        assert_eq!(trie.get(b"hello"), None);
        assert_eq!(trie.get(b""), None);
        assert!(!trie.contains_prefix(b""));
        trie.insert(b"hello", 1);
        assert_eq!(trie.delete(b"hello"), Some(1));
        assert_eq!(trie.delete(b"hello "), None);
        trie.insert(b"hello", 1);
        trie.insert(b"hell", 2);
        trie.insert(b"hello world", 3);
        assert_eq!(trie.get(b"hello"), Some(&1));
        assert_eq!(trie.get(b"hell"), Some(&2));
        assert_eq!(trie.get(b"hello world"), Some(&3));
        assert_eq!(trie.contains_prefix(b"hell"), true);
        assert_eq!(trie.contains_prefix(b"hello"), true);
        assert_eq!(trie.contains_prefix(b""), true);
        assert_eq!(trie.contains_prefix(b"hello world!"), false);
        assert_eq!(trie.contains_prefix(b"test"), false);
        assert_eq!(trie.delete(b"hello"), Some(1));
        assert_eq!(trie.get(b"hello"), None);
        let stats = trie.stats();
        assert_eq!(stats.num_keys, 2);

        let dir = env!("CARGO_MANIFEST_DIR");
        let index = fs::read_to_string(PathBuf::from(dir).join("resources/test/index.txt"))
            .expect("failed to read file");
        let n = 100_000;
        let words: Vec<_> = index.lines().map(|s| s.as_bytes()).take(n).collect();

        let mut trie: Trie<_> = words.iter().enumerate().map(|(i, w)| (w, i)).collect();
        let stats = trie.stats();
        println!("{:#?}", stats);
        assert_eq!(stats.num_keys, n);
        for (i, word) in words.iter().enumerate() {
            assert_eq!(trie.get(word), Some(&i));
            for j in 0..word.len() {
                assert!(trie.contains_prefix(&word[..=j]));
            }
        }
        for (i, word) in words.iter().enumerate() {
            let even = i % 2 == 0;
            if even {
                assert_eq!(trie.delete(word), Some(i));
                assert_eq!(trie.get(word), None);
            } else {
                assert_eq!(trie.get(word), Some(&i));
            }
        }
        let stats = trie.stats();
        assert_eq!(stats.num_keys, n / 2);
    }
}
