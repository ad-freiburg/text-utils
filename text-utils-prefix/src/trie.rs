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
    num_keys: usize,
}

impl<V> Trie<V> {
    pub fn size(&self) -> usize {
        self.num_keys
    }
}

impl<V> Default for Trie<V> {
    fn default() -> Self {
        Self {
            root: None,
            num_keys: 0,
        }
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
        if node.value.is_none() {
            self.num_keys += 1;
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
        self.num_keys -= 1;
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
        assert_eq!(trie.size(), 2);

        let dir = env!("CARGO_MANIFEST_DIR");
        let index = fs::read_to_string(PathBuf::from(dir).join("resources/test/index.txt"))
            .expect("failed to read file");
        let n = 100_000;
        let words: Vec<_> = index.lines().map(|s| s.as_bytes()).take(n).collect();

        let mut trie: Trie<_> = words.iter().enumerate().map(|(i, w)| (w, i)).collect();
        assert_eq!(trie.size(), n);
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
        assert_eq!(trie.size(), n / 2);
    }
}
