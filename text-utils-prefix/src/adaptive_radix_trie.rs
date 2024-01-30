use std::{
    collections::HashMap,
    iter::{empty, once},
};

use crate::{ContinuationSearch, PrefixSearch};

type Index<const N: usize> = [u8; N];
type Children<V, const N: usize> = [Option<Box<Node<V>>>; N];

#[derive(Default, Debug)]
enum NodeType<V> {
    #[default]
    Empty,
    Leaf(V),
    N4(Index<4>, Children<V, 4>, usize),
    N16(Index<16>, Children<V, 16>, usize),
    N48(Box<Index<256>>, Children<V, 48>, usize),
    N256(Children<V, 256>, usize),
}

#[derive(Debug)]
struct Node<V> {
    prefix: Box<[u8]>,
    inner: NodeType<V>,
}

#[derive(Debug)]
pub struct AdaptiveRadixTrie<V> {
    root: Option<Node<V>>,
}

#[derive(Debug)]
pub struct AdaptiveRadixTrieStats {
    pub depth: usize,
    pub num_nodes: usize,
    pub num_keys: usize,
    pub node_info: HashMap<String, (usize, f32)>,
}

impl<V> AdaptiveRadixTrie<V> {
    pub fn stats(&self) -> AdaptiveRadixTrieStats {
        let mut dist = HashMap::from_iter(
            ["leaf", "n4", "n16", "n48", "n256"]
                .iter()
                .map(|&s| (s.to_string(), (0, 0.0))),
        );
        let Some(root) = &self.root else {
            return AdaptiveRadixTrieStats {
                depth: 0,
                num_nodes: 0,
                num_keys: 0,
                node_info: dist,
            };
        };
        let mut stack = vec![(root, 0)];
        let mut max_depth = 0;
        while let Some((node, depth)) = stack.pop() {
            max_depth = max_depth.max(depth);
            let name = match &node.inner {
                NodeType::Empty => unreachable!("should not happen"),
                NodeType::Leaf(_) => "leaf",
                NodeType::N4(..) => "n4",
                NodeType::N16(..) => "n16",
                NodeType::N48(..) => "n48",
                NodeType::N256(..) => "n256",
            };
            let val = dist.get_mut(name).unwrap();
            val.0 += 1;
            let n = val.0 as f32;
            val.1 = (val.1 * (n - 1.0) + node.prefix.len() as f32) / n;
            stack.extend(node.children().map(|child| (child, depth + 1)));
        }
        AdaptiveRadixTrieStats {
            depth: max_depth,
            num_nodes: dist.iter().map(|(_, (n, _))| n).sum(),
            num_keys: dist["leaf"].0,
            node_info: dist,
        }
    }
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
    FullPrefix(u8),
    Exact,
    Partial(usize, u8),
}

impl<V> Node<V> {
    fn new_leaf(prefix: Vec<u8>, value: V) -> Self {
        Self {
            prefix: prefix.into_boxed_slice(),
            inner: NodeType::Leaf(value),
        }
    }

    fn new_inner(prefix: Vec<u8>) -> Self {
        Self {
            prefix: prefix.into_boxed_slice(),
            inner: NodeType::N4(std::array::from_fn(|_| 0), std::array::from_fn(|_| None), 0),
        }
    }

    #[inline]
    fn is_leaf(&self) -> bool {
        matches!(self.inner, NodeType::Leaf(_))
    }

    #[inline]
    fn is_inner(&self) -> bool {
        !self.is_leaf()
    }

    #[inline]
    fn matching(&self, key: &mut impl Iterator<Item = u8>, offset: usize) -> Matching {
        let mut i = offset;
        while i < self.prefix.len() {
            let Some(k) = key.next() else {
                return Matching::FullKey(i);
            };
            if k != self.prefix[i] {
                return Matching::Partial(i, k);
            }
            i += 1;
        }
        if let Some(k) = key.next() {
            Matching::FullPrefix(k)
        } else {
            Matching::Exact
        }
    }

    #[inline]
    fn find_iter(&self, mut key: impl Iterator<Item = u8>) -> Option<&Self> {
        let mut node = self;
        loop {
            if node.is_leaf() {
                if let Matching::Exact = node.matching(&mut key, 0) {
                    return Some(node);
                }
                break;
            }

            let Matching::FullPrefix(k) = node.matching(&mut key, 0) else {
                // if we dont match the full node prefix,
                // we can return early
                return None;
            };

            let Some(child) = node.find_child(k) else {
                break;
            };
            node = child;
        }
        None
    }

    #[inline]
    fn has_child(&self, key: u8) -> bool {
        self.find_child(key).is_some()
    }

    fn children(&self) -> Box<dyn Iterator<Item = &Self> + '_> {
        match &self.inner {
            NodeType::Empty | NodeType::Leaf(_) => Box::new(empty()),
            NodeType::N4(_, children, num_children) => Box::new(
                children[..*num_children]
                    .iter()
                    .filter_map(|child| child.as_deref()),
            ),
            NodeType::N16(_, children, num_children) => Box::new(
                children[..*num_children]
                    .iter()
                    .filter_map(|child| child.as_deref()),
            ),
            NodeType::N48(_, children, num_children) => Box::new(
                children[..*num_children]
                    .iter()
                    .filter_map(|child| child.as_deref()),
            ),
            NodeType::N256(children, _) => {
                Box::new(children.iter().filter_map(|child| child.as_deref()))
            }
        }
    }

    #[inline]
    fn set_child(&mut self, key: u8, child: Self) {
        // potentially upgrade the current node before insertion, will change
        // nothing if the node does not need to be upgraded
        assert!(self.find_child(key).is_none());
        self.upgrade();
        match &mut self.inner {
            NodeType::Empty | NodeType::Leaf(_) => unreachable!("should not happen"),
            NodeType::N4(keys, children, num_children) => {
                // also keep sorted order for n4 for easier upgrade
                let idx = keys[..*num_children].binary_search(&key).unwrap_err();
                if idx < *num_children {
                    keys[idx..].rotate_right(1);
                    children[idx..].rotate_right(1);
                }
                keys[idx] = key;
                children[idx] = Some(Box::new(child));
                *num_children += 1;
            }
            NodeType::N16(keys, children, num_children) => {
                let idx = keys[..*num_children].binary_search(&key).unwrap_err();
                if idx < *num_children {
                    keys[idx..].rotate_right(1);
                    children[idx..].rotate_right(1);
                }
                keys[idx] = key;
                children[idx] = Some(Box::new(child));
                *num_children += 1;
            }
            NodeType::N48(index, children, num_children) => {
                index[key as usize] = *num_children as u8;
                children[*num_children] = Some(Box::new(child));
                *num_children += 1;
            }
            NodeType::N256(children, num_children) => {
                children[key as usize] = Some(Box::new(child));
                *num_children += 1;
            }
        }
    }

    #[inline]
    fn remove_child(&mut self, key: u8) {
        // potentially downgrade the current node before removal, will change
        // nothing if the node does not need to be downgraded
        self.downgrade();
        match &mut self.inner {
            NodeType::Empty | NodeType::Leaf(_) => unreachable!("should not happen"),
            NodeType::N4(_, _, _) => todo!(),
            NodeType::N16(_, _, _) => todo!(),
            NodeType::N48(_, _, _) => todo!(),
            NodeType::N256(_, _) => todo!(),
        }
    }

    #[inline]
    fn contains_prefix_iter(
        &self,
        mut key: impl Iterator<Item = u8>,
        offset: usize,
    ) -> Option<(&Self, usize)> {
        let mut node = self;
        // extend given key with null byte
        // because its needed for the correctness of the algorithm
        // when it comes to key lookup
        loop {
            let k = match node.matching(&mut key, offset) {
                Matching::FullKey(n) => return Some((node, n)),
                Matching::Exact => return Some((node, node.prefix.len())),
                Matching::FullPrefix(k) => k,
                Matching::Partial(..) => break,
            };

            let Some(child) = node.find_child(k) else {
                break;
            };
            node = child;
        }
        None
    }

    #[inline]
    fn find_child(&self, key: u8) -> Option<&Self> {
        match &self.inner {
            NodeType::Empty | NodeType::Leaf(_) => None,
            NodeType::N4(keys, children, num_children) => {
                for i in 0..*num_children {
                    if keys[i] == key {
                        return children[i].as_deref();
                    }
                }
                None
            }
            NodeType::N16(keys, children, num_children) => {
                let idx = keys[..*num_children].binary_search(&key).ok()?;
                children[idx].as_deref()
            }
            NodeType::N48(keys, children, _) => {
                children.get(keys[key as usize] as usize)?.as_deref()
            }
            NodeType::N256(children, _) => children[key as usize].as_deref(),
        }
    }

    #[inline]
    fn find_child_mut(&mut self, key: u8) -> Option<&mut Self> {
        match &mut self.inner {
            NodeType::Empty | NodeType::Leaf(_) => None,
            NodeType::N4(keys, children, num_children) => {
                for i in 0..*num_children {
                    if keys[i] == key {
                        return children[i].as_deref_mut();
                    }
                }
                None
            }
            NodeType::N16(keys, children, num_children) => {
                let idx = keys[..*num_children].binary_search(&key).ok()?;
                children[idx].as_deref_mut()
            }
            NodeType::N48(keys, children, _) => children
                .get_mut(keys[key as usize] as usize)?
                .as_deref_mut(),
            NodeType::N256(children, _) => children[key as usize].as_deref_mut(),
        }
    }

    fn upgrade(&mut self) {
        let inner = match &mut self.inner {
            NodeType::Empty | NodeType::Leaf(_) => {
                unreachable!("should not happen")
            }
            NodeType::N256(_, num_children) => {
                // upgrade should only be called on non empty n256 nodes
                assert!(*num_children < 256);
                return;
            }
            NodeType::N4(keys, children, num_children) => {
                if *num_children < 4 {
                    // nothing to do
                    return;
                }
                assert_eq!(*num_children, 4);
                // just move over because n4 is also sorted
                NodeType::N16(
                    std::array::from_fn(|i| if i < 4 { keys[i] } else { 0 }),
                    std::array::from_fn(|i| {
                        if i < 4 {
                            assert!(children[i].is_some());
                            std::mem::take(&mut children[i])
                        } else {
                            None
                        }
                    }),
                    4,
                )
            }
            NodeType::N16(keys, children, num_children) => {
                if *num_children < 16 {
                    // nothing to do
                    return;
                }
                assert_eq!(*num_children, 16);
                let mut index = [u8::MAX; 256];
                for (i, k) in keys.iter().enumerate() {
                    index[*k as usize] = i as u8;
                }
                NodeType::N48(
                    Box::new(index),
                    std::array::from_fn(|i| {
                        if i < 16 {
                            assert!(children[i].is_some());
                            std::mem::take(&mut children[i])
                        } else {
                            None
                        }
                    }),
                    16,
                )
            }
            NodeType::N48(index, children, num_children) => {
                if *num_children < 48 {
                    // nothing to do
                    return;
                }
                assert_eq!(*num_children, 48);
                NodeType::N256(
                    std::array::from_fn(|i| {
                        let idx = index[i];
                        if idx < 48 {
                            assert!(children[idx as usize].is_some());
                            std::mem::take(&mut children[idx as usize])
                        } else {
                            None
                        }
                    }),
                    48,
                )
            }
        };
        self.inner = inner;
    }

    fn downgrade(&mut self) {
        todo!()
    }
}

impl<V> PrefixSearch for AdaptiveRadixTrie<V> {
    type Value = V;

    fn insert<K>(&mut self, key: K, value: V)
    where
        K: AsRef<[u8]>,
    {
        let mut key = key.as_ref().iter().copied().chain(once(0));
        // empty tree
        let Some(root) = &mut self.root else {
            // insert leaf at root
            self.root = Some(Node::new_leaf(key.collect(), value));
            return;
        };
        let mut node = root;
        loop {
            let matching = node.matching(&mut key, 0);
            if node.is_leaf() {
                let (inner_prefix, new_prefix, n, k) = match matching {
                    Matching::FullKey(_) => unreachable!("should not happen"),
                    Matching::FullPrefix(_) => unreachable!("should not happen"),
                    Matching::Partial(n, k) => (
                        node.prefix[..n].to_vec(),
                        node.prefix[n + 1..].to_vec(),
                        n,
                        k,
                    ),
                    Matching::Exact => {
                        // exact match, only replace leaf value
                        node.inner = NodeType::Leaf(value);
                        return;
                    }
                };
                let mut inner = Node::new_inner(inner_prefix);
                let NodeType::Leaf(node_value) = std::mem::take(&mut node.inner) else {
                    unreachable!("should not happen");
                };
                inner.set_child(node.prefix[n], Node::new_leaf(new_prefix, node_value));
                inner.set_child(k, Node::new_leaf(key.collect(), value));
                *node = inner;
                break;
            } else if let Matching::FullPrefix(k) = matching {
                // full prefix match, either go to next child
                // or append leaf with rest of key
                if node.has_child(k) {
                    node = node.find_child_mut(k).expect("should not happen");
                    continue;
                }
                node.set_child(k, Node::new_leaf(key.collect(), value));
            } else if let Matching::Partial(n, k) = matching {
                // partial prefix match, introduce new inner node
                let mut inner = Node::new_inner(node.prefix[..n].to_vec());
                let mut new_node = Node::new_inner(node.prefix[n + 1..].to_vec());
                new_node.inner = std::mem::take(&mut node.inner);
                inner.set_child(node.prefix[n], new_node);
                inner.set_child(k, Node::new_leaf(key.collect(), value));
                *node = inner;
            }
            break;
        }
    }

    fn delete<K>(&mut self, key: K) -> Option<V>
    where
        K: AsRef<[u8]>,
    {
        let Some(root) = &mut self.root else {
            return None;
        };

        // handle special case where root is leaf
        if root.is_leaf() {
            let NodeType::Leaf(value) = std::mem::take(&mut root.inner) else {
                unreachable!("should not happen");
            };
            self.root = None;
            return Some(value);
        }

        let mut node = root;
        let mut key = key.as_ref().iter().copied().chain(once(0));
        loop {
            let matching = node.matching(&mut key, 0);

            let Matching::FullPrefix(k) = matching else {
                // on inner nodes we always need full prefix matching
                return None;
            };

            // return if we dont find a child
            let child = node.find_child(k)?;

            // traverse down if child is inner
            if child.is_inner() {
                node = node.find_child_mut(k)?;
                continue;
            }

            // handle case if child is leaf
            let Matching::Exact = child.matching(&mut key, 0) else {
                break;
            };

            todo!();
            // key is an exact match for a leaf
            // let NodeType::Inner(children) = &mut node.inner else {
            //     unreachable!("should not happen");
            // };
            // let child = std::mem::take(&mut children[k as usize])?;
            // let NodeType::Leaf(value) = child.inner else {
            //     unreachable!("should not happen");
            // };
            // let child_indices: Vec<_> = children
            //     .iter()
            //     .enumerate()
            //     .filter_map(|(i, child)| child.as_ref().map(|_| i))
            //     .collect();
            // assert!(!child_indices.is_empty());
            // if child_indices.len() == 1 {
            //     // if we only have one child left, we can merge
            //     // the child into the current node
            //     let single_child_k = child_indices.into_iter().next().unwrap();
            //     let single_child = std::mem::take(&mut children[single_child_k])?;
            //     let new_prefix: Vec<_> = node
            //         .prefix
            //         .iter()
            //         .copied()
            //         .chain(once(single_child_k as u8))
            //         .chain(single_child.prefix.iter().copied())
            //         .collect();
            //     node.prefix = new_prefix.into_boxed_slice();
            //     node.inner = single_child.inner;
            // }
            // return Some(value);
        }
        None
    }

    fn get<K>(&self, key: K) -> Option<&V>
    where
        K: AsRef<[u8]>,
    {
        let Some(root) = &self.root else {
            return None;
        };

        let key = key.as_ref().iter().copied().chain(once(0));
        root.find_iter(key).and_then(|node| match &node.inner {
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

        let key = prefix.as_ref().iter().copied();
        root.contains_prefix_iter(key, 0).is_some()
    }
}

impl<V> ContinuationSearch for AdaptiveRadixTrie<V> {
    fn continuations<'a, P>(&'a self, prefix: P) -> impl Iterator<Item = (Vec<u8>, &'a V)>
    where
        P: AsRef<[u8]>,
        V: 'a,
    {
        empty()
    }

    fn contains_continuation<P, C>(&self, prefix: P, continuation: C) -> bool
    where
        P: AsRef<[u8]>,
        C: AsRef<[u8]>,
    {
        let Some(root) = &self.root else {
            return false;
        };

        let key = prefix
            .as_ref()
            .iter()
            .chain(continuation.as_ref().iter())
            .copied();
        root.contains_prefix_iter(key, 0).is_some()
    }

    fn contains_continuations<P, C>(&self, prefix: P, continuations: &[C]) -> Vec<usize>
    where
        P: AsRef<[u8]>,
        C: AsRef<[u8]>,
    {
        let Some(root) = &self.root else {
            return vec![];
        };

        let key = prefix.as_ref().iter().copied();
        let Some((node, n)) = root.contains_prefix_iter(key, 0) else {
            return vec![];
        };

        continuations
            .iter()
            .enumerate()
            .filter_map(|(i, c)| {
                let key = c.as_ref().iter().copied();
                if node.contains_prefix_iter(key, n).is_some() {
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
        let mut result = vec![];
        let Some(root) = &self.root else {
            return result;
        };

        let key = prefix.as_ref().iter().copied();
        let Some((node, n)) = root.contains_prefix_iter(key, 0) else {
            return result;
        };

        let mut i = 0;
        while let Some(&j) = permutation.get(i) {
            let continuation = continuations[j].as_ref();
            if node
                .contains_prefix_iter(continuation.iter().copied(), n)
                .is_some()
            {
                result.push(j);
            } else {
                i += skips[i];
            }
            i += 1;
        }

        result
    }
}

#[cfg(test)]
mod test {
    use crate::{adaptive_radix_trie::AdaptiveRadixTrie, PrefixSearch};
    use std::fs;
    use std::path::PathBuf;

    #[test]
    fn test_trie() {
        let mut trie = AdaptiveRadixTrie::default();
        assert_eq!(trie.get(b"hello"), None);
        assert_eq!(trie.get(b""), None);
        assert!(!trie.contains_prefix(b""));
        trie.insert(b"hello", 1);
        // assert_eq!(trie.delete(b"hello"), Some(1));
        // assert_eq!(trie.delete(b"hello "), None);
        // trie.insert(b"hello", 1);
        trie.insert(b"hell", 2);
        trie.insert(b"hello world", 3);
        // println!("{:#?}", trie);
        assert_eq!(trie.get(b"hello"), Some(&1));
        assert_eq!(trie.get(b"hell"), Some(&2));
        assert_eq!(trie.get(b"hello world"), Some(&3));
        assert_eq!(trie.contains_prefix(b"hell"), true);
        assert_eq!(trie.contains_prefix(b"hello"), true);
        assert_eq!(trie.contains_prefix(b""), true);
        assert_eq!(trie.contains_prefix(b"hello world!"), false);
        assert_eq!(trie.contains_prefix(b"test"), false);
        // assert_eq!(trie.delete(b"hello"), Some(1));
        // assert_eq!(trie.get(b"hello"), None);
        // assert_eq!(trie.size(), 2);

        let dir = env!("CARGO_MANIFEST_DIR");
        let index = fs::read_to_string(PathBuf::from(dir).join("resources/test/index.txt"))
            .expect("failed to read file");
        let n = 100_000;
        let words: Vec<_> = index.lines().map(|s| s.as_bytes()).take(n).collect();

        let mut trie: AdaptiveRadixTrie<_> =
            words.iter().enumerate().map(|(i, w)| (w, i)).collect();
        let stats = trie.stats();
        assert_eq!(stats.num_keys, n);
        for (i, word) in words.iter().enumerate() {
            assert_eq!(trie.get(word), Some(&i));
            for j in 0..word.len() {
                assert!(trie.contains_prefix(&word[..=j]));
            }
        }
        println!("{:#?}", trie.stats());
        // for (i, word) in words.iter().enumerate() {
        //     let even = i % 2 == 0;
        //     if even {
        //         assert_eq!(trie.delete(word), Some(i));
        //         assert_eq!(trie.get(word), None);
        //     } else {
        //         assert_eq!(trie.get(word), Some(&i));
        //     }
        // }
        // let stats = trie.stats();
        // assert_eq!(stats.num_keys, n / 2);
    }
}
