use std::iter::once;

use crate::PrefixSearch;

type Index<const N: usize> = [u8; N];
type Children<V, const N: usize> = [Option<Box<Node<V>>>; N];

enum NodeType<V> {
    Leaf(V),
    N4(Index<4>, Children<V, 4>, usize),
    N16(Index<16>, Children<V, 16>, usize),
    N48(Box<Index<256>>, Children<V, 48>, usize),
    N256(Children<V, 256>, usize),
}

struct Node<V> {
    prefix: Box<[u8]>,
    inner: NodeType<V>,
}

pub struct AdaptiveRadixTrie<V> {
    root: Option<Node<V>>,
}

impl<V> Default for AdaptiveRadixTrie<V> {
    fn default() -> Self {
        Self { root: None }
    }
}

impl<K, V> FromIterator<(K, V)> for AdaptiveRadixTrie<V>
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

enum Matching {
    FullKey(usize),
    FullNode,
    Partial(usize, u8),
}

impl<V> Node<V> {
    #[inline]
    fn is_leaf(&self) -> bool {
        matches!(self.inner, NodeType::Leaf(_))
    }

    #[inline]
    fn advance_key<'a>(&self, key: &mut impl Iterator<Item = &'a u8>) -> Matching {
        let mut i = 0;
        while i < self.prefix.len() {
            let Some(k) = key.next() else {
                return Matching::FullKey(i);
            };
            if k != &self.prefix[i] {
                return Matching::Partial(i, *k);
            }
            i += 1;
        }
        Matching::FullNode
    }

    #[inline]
    fn exact_match<'a>(&self, key: &mut impl Iterator<Item = &'a u8>) -> bool {
        let mut i = 0;
        while i < self.prefix.len() {
            let Some(k) = key.next() else {
                return false;
            };
            if k != &self.prefix[i] {
                return false;
            }
            i += 1;
        }
        // we have to be at the end of the key for an exact match
        key.next().is_none()
    }

    #[inline]
    fn find(&self, key: &[u8]) -> Option<&Self> {
        let mut node = self;
        // extend given key with null byte
        // because its needed for the correctness of the algorithm
        // when it comes to key lookup
        let mut key = key.iter().chain(once(&0));
        loop {
            if node.is_leaf() {
                if self.exact_match(&mut key) {
                    return Some(node);
                }
                break;
            }

            let Matching::FullNode = self.advance_key(&mut key) else {
                // if we have not a full node match,
                // we can return early
                return None;
            };

            let k = key.next()?;
            let Some(child) = node.find_child(k) else {
                break;
            };
            node = child;
        }
        None
    }

    #[inline]
    fn find_child(&self, key: &u8) -> Option<&Self> {
        match &self.inner {
            NodeType::Leaf(_) => None,
            NodeType::N4(keys, children, num_children) => {
                for i in 0..*num_children {
                    if &keys[i] == key {
                        return children[i].as_deref();
                    }
                }
                None
            }
            NodeType::N16(keys, children, num_children) => {
                let idx = keys[..*num_children].binary_search(key).ok()?;
                children[idx].as_deref()
            }
            NodeType::N48(keys, children, _) => {
                children.get(keys[*key as usize] as usize)?.as_deref()
            }
            NodeType::N256(children, _) => children[*key as usize].as_deref(),
        }
    }

    fn upgrade(self) -> Result<Self, Self> {
        todo!()
    }

    fn downgrade(self) -> Result<Self, Self> {
        todo!()
    }
}

impl<V> PrefixSearch<V> for AdaptiveRadixTrie<V> {
    fn insert<K>(&mut self, key: K, value: V)
    where
        K: AsRef<[u8]>,
    {
        todo!()
    }

    fn delete<K>(&mut self, key: K) -> Option<V>
    where
        K: AsRef<[u8]>,
    {
        todo!()
    }

    fn get<K>(&self, key: K) -> Option<&V>
    where
        K: AsRef<[u8]>,
    {
        let Some(root) = &self.root else {
            return None;
        };

        root.find(key.as_ref()).and_then(|node| match &node.inner {
            NodeType::Leaf(v) => Some(v),
            _ => None,
        })
    }

    fn contains_prefix<P>(&self, prefix: P) -> bool
    where
        P: AsRef<[u8]>,
    {
        let Some(root) = &self.root else {
            return false;
        };

        todo!();
    }
}
