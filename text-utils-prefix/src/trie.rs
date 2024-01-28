use crate::PrefixSearch;

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

pub struct Trie<V> {
    root: Option<Node<V>>,
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
}

impl<V> PrefixSearch<V> for Trie<V> {
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

    #[test]
    fn test_trie() {
        let mut trie = Trie::default();
        assert_eq!(trie.get(b"hello"), None);
        trie.insert(b"hello", 1);
        assert_eq!(trie.get(b"hello"), Some(&1));
        assert_eq!(trie.contains_prefix(b"hell"), true);
        assert_eq!(trie.contains_prefix(b"hello"), true);
        assert_eq!(trie.contains_prefix(b""), true);
        assert_eq!(trie.contains_prefix(b"hello world"), false);
        assert_eq!(trie.contains_prefix(b"test"), false);
        assert_eq!(trie.delete(b"hello"), Some(1));
        assert_eq!(trie.get(b"hello"), None);
    }
}
