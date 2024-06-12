use std::{
    collections::HashMap,
    iter::{empty, once},
};

use crate::PrefixSearch;

#[derive(Debug)]
enum NodeType<V> {
    Leaf(V),
    Inner(Box<[Option<Box<Node<V>>>; 256]>),
}

#[derive(Debug)]
struct Node<V> {
    prefix: Box<[u8]>,
    inner: NodeType<V>,
}

#[derive(Debug)]
pub struct PatriciaTrie<V> {
    root: Option<Node<V>>,
}

#[derive(Debug, Default)]
pub struct PatriciaTrieStats {
    pub max_depth: usize,
    pub avg_depth: f32,
    pub num_nodes: usize,
    pub num_keys: usize,
    pub node_info: HashMap<String, (usize, f32)>,
}

impl<V> PatriciaTrie<V> {
    pub fn stats(&self) -> PatriciaTrieStats {
        let mut dist =
            HashMap::from_iter(["leaf", "inner"].iter().map(|&s| (s.to_string(), (0, 0.0))));
        let Some(root) = &self.root else {
            return PatriciaTrieStats::default();
        };
        let mut stack = vec![(root, 0)];
        let mut max_depth = 0;
        let mut avg_depth = (0, 0.0);
        while let Some((node, depth)) = stack.pop() {
            max_depth = max_depth.max(depth);
            let name = match &node.inner {
                NodeType::Leaf(_) => "leaf",
                NodeType::Inner(..) => "inner",
            };
            let val = dist.get_mut(name).expect("should not happen");
            val.0 += 1;
            let n = val.0 as f32;
            val.1 = (val.1 * (n - 1.0) + node.prefix.len() as f32) / n;
            avg_depth.0 += 1;
            let n = avg_depth.0 as f32;
            avg_depth.1 = (avg_depth.1 * (n - 1.0) + depth as f32) / n;
            stack.extend(node.children().map(|(_, child)| (child, depth + 1)));
        }
        PatriciaTrieStats {
            max_depth,
            avg_depth: avg_depth.1,
            num_nodes: dist.iter().map(|(_, (n, _))| n).sum(),
            num_keys: dist["leaf"].0,
            node_info: dist,
        }
    }
}

impl<V> Default for PatriciaTrie<V> {
    fn default() -> Self {
        Self { root: None }
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
            inner: NodeType::Inner(Box::new(std::array::from_fn(|_| None))),
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
        match &self.inner {
            NodeType::Leaf(_) => false,
            NodeType::Inner(children) => children[key as usize].is_some(),
        }
    }

    fn children(&self) -> Box<dyn Iterator<Item = (u8, &Self)> + '_> {
        match &self.inner {
            NodeType::Leaf(_) => Box::new(empty()),
            NodeType::Inner(children) => Box::new(
                children
                    .iter()
                    .enumerate()
                    .filter_map(|(i, child)| child.as_ref().map(|child| (i as u8, child.as_ref()))),
            ),
        }
    }

    #[inline]
    fn find_child(&self, key: u8) -> Option<&Self> {
        match &self.inner {
            NodeType::Leaf(_) => None,
            NodeType::Inner(children) => children[key as usize].as_deref(),
        }
    }

    #[inline]
    fn find_child_mut(&mut self, key: u8) -> Option<&mut Self> {
        match &mut self.inner {
            NodeType::Leaf(_) => None,
            NodeType::Inner(children) => children[key as usize].as_deref_mut(),
        }
    }

    #[inline]
    fn set_child(&mut self, key: u8, child: Self) {
        let NodeType::Inner(children) = &mut self.inner else {
            unreachable!("set child called on leaf node");
        };
        let pos = &mut children[key as usize];
        if pos.is_some() {
            unreachable!("should not happen");
        }
        *pos = Some(Box::new(child));
    }

    #[inline]
    fn contains_prefix_iter(
        &self,
        mut key: impl Iterator<Item = u8>,
        mut offset: usize,
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
            // reset offset after first node
            offset = 0;

            let Some(child) = node.find_child(k) else {
                break;
            };
            node = child;
        }
        None
    }

    #[inline]
    fn leaves(&self, mut path: Vec<u8>) -> Box<dyn Iterator<Item = (Vec<u8>, &V)> + '_> {
        path.extend(self.prefix.iter().copied());
        if let NodeType::Leaf(value) = &self.inner {
            // dont keep last element (null byte) for full paths
            path.pop();
            return Box::new(once((path, value)));
        }
        Box::new(self.children().flat_map(move |(k, child)| {
            let mut key = path.clone();
            key.push(k);
            child.leaves(key)
        }))
    }
}

impl<V> PrefixSearch for PatriciaTrie<V> {
    type Value = V;

    fn insert(&mut self, key: &[u8], value: V) -> Option<V> {
        let mut key = key.iter().copied().chain(once(0));
        // empty tree
        let Some(root) = &mut self.root else {
            // insert leaf at root
            self.root = Some(Node::new_leaf(key.collect(), value));
            return None;
        };
        let mut node = root;
        loop {
            match node.matching(&mut key, 0) {
                Matching::FullKey(_) => unreachable!("should not happen"),
                Matching::FullPrefix(k) => {
                    // full prefix match, either go to next child
                    // or append leaf with rest of key
                    if node.has_child(k) {
                        node = node.find_child_mut(k).unwrap();
                        continue;
                    }
                    node.set_child(k, Node::new_leaf(key.collect(), value));
                    break;
                }
                Matching::Partial(n, k) => {
                    let inner_prefix = node.prefix[..n].to_vec();
                    let old_prefix = node.prefix[n + 1..].to_vec();
                    let p_k = node.prefix[n];

                    let mut old_node = std::mem::replace(node, Node::new_inner(inner_prefix));
                    old_node.prefix = old_prefix.into();
                    node.set_child(k, Node::new_leaf(key.collect(), value));
                    node.set_child(p_k, old_node);
                    break;
                }
                Matching::Exact => {
                    // exact match, only replace leaf value
                    let NodeType::Leaf(node_value) =
                        std::mem::replace(&mut node.inner, NodeType::Leaf(value))
                    else {
                        unreachable!("should not happen");
                    };
                    return Some(node_value);
                }
            };
        }
        None
    }

    fn delete(&mut self, key: &[u8]) -> Option<V> {
        let root = self.root.as_mut()?;

        // handle special case where root is leaf
        if root.is_leaf() {
            let Some(Node {
                inner: NodeType::Leaf(value),
                ..
            }) = self.root.take()
            else {
                unreachable!("should not happen");
            };
            return Some(value);
        }

        let mut node = root;
        let mut key = key.iter().copied().chain(once(0));
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
            // key is an exact match for a leaf
            let NodeType::Inner(children) = &mut node.inner else {
                unreachable!("should not happen");
            };
            let child = children[k as usize].take().unwrap();
            let NodeType::Leaf(value) = child.inner else {
                unreachable!("should not happen");
            };
            let child_indices: Vec<_> = children
                .iter()
                .enumerate()
                .filter_map(|(i, child)| child.as_ref().map(|_| i))
                .collect();
            assert!(!child_indices.is_empty());
            if child_indices.len() == 1 {
                // if we only have one child left, we can merge
                // the child into the current node
                let single_child_k = child_indices.into_iter().next().unwrap();
                let single_child = children[single_child_k].take().unwrap();
                let new_prefix: Vec<_> = node
                    .prefix
                    .iter()
                    .copied()
                    .chain(once(single_child_k as u8))
                    .chain(single_child.prefix.iter().copied())
                    .collect();
                node.prefix = new_prefix.into_boxed_slice();
                node.inner = single_child.inner;
            }
            return Some(value);
        }
        None
    }

    fn get(&self, key: &[u8]) -> Option<&V> {
        let root = &self.root.as_ref()?;

        let key = key.iter().copied().chain(once(0));
        root.find_iter(key).and_then(|node| match &node.inner {
            NodeType::Leaf(v) => Some(v),
            _ => None,
        })
    }

    fn contains(&self, prefix: &[u8]) -> bool {
        let Some(root) = &self.root else {
            return false;
        };

        let key = prefix.iter().copied();
        root.contains_prefix_iter(key, 0).is_some()
    }

    fn values_along_path(&self, prefix: &[u8]) -> Vec<(usize, &Self::Value)> {
        let Some(root) = &self.root else {
            return vec![];
        };

        let mut path = vec![];
        let mut node = root;
        let mut key = prefix.iter().copied();
        let mut i = 0;
        loop {
            match node.matching(&mut key, 0) {
                Matching::FullKey(_) => break,
                Matching::FullPrefix(k) => {
                    i += node.prefix.len();
                    if let Some(leaf) = node.find_child(0) {
                        let NodeType::Leaf(v) = &leaf.inner else {
                            unreachable!("should not happen");
                        };
                        path.push((i, v));
                    }
                    let Some(child) = node.find_child(k) else {
                        break;
                    };
                    node = child;
                    i += 1;
                }
                Matching::Exact => {
                    let Some(child) = node.find_child(0) else {
                        break;
                    };
                    let NodeType::Leaf(v) = &child.inner else {
                        unreachable!("should not happen");
                    };
                    path.push((i + node.prefix.len(), v));
                    break;
                }
                Matching::Partial(..) => break,
            };
        }
        path
    }

    fn continuations(&self, prefix: &[u8]) -> Box<dyn Iterator<Item = (Vec<u8>, &V)> + '_> {
        let Some(root) = &self.root else {
            return Box::new(empty());
        };
        let mut node = root;
        let mut key = prefix.iter().copied();
        let mut prefix = vec![];
        loop {
            let k = match node.matching(&mut key, 0) {
                Matching::FullKey(_) | Matching::Exact => {
                    break;
                }
                Matching::FullPrefix(k) => {
                    prefix.extend(node.prefix.iter().copied());
                    k
                }
                Matching::Partial(..) => return Box::new(empty()),
            };

            let Some(child) = node.find_child(k) else {
                return Box::new(empty());
            };
            prefix.push(k);
            node = child;
        }

        node.leaves(prefix)
    }
}

impl<K, V> FromIterator<(K, V)> for PatriciaTrie<V>
where
    K: AsRef<[u8]>,
{
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let mut trie = PatriciaTrie::default();
        for (key, value) in iter {
            trie.insert(key.as_ref(), value);
        }
        trie
    }
}
