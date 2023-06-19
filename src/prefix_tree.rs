use std::collections::BTreeMap;

use crate::{
    prefix::PrefixTreeSearch,
    unicode::{normalize, Normalization},
};

pub trait PrefixTreeNode {
    type Value;

    fn get_value(&self) -> Option<&Self::Value>;

    fn set_value(&mut self, value: Self::Value);

    fn get_child(&self, key: &u8) -> Option<&Self>;

    fn get_children(&self) -> Box<dyn Iterator<Item = (&u8, &Self)> + '_>;

    fn set_child(&mut self, key: &u8, value: Self);

    fn get_child_mut(&mut self, key: &u8) -> Option<&mut Self>;

    #[inline]
    fn find(&self, key: &[u8]) -> Option<&Self> {
        let mut node = self;
        for idx in key {
            if let Some(child) = node.get_child(idx) {
                node = child;
            } else {
                return None;
            }
        }
        Some(node)
    }

    #[inline]
    fn find_continuations(&self) -> Box<dyn Iterator<Item = (Vec<u8>, &Self::Value)> + '_> {
        Box::new(self.get_children().flat_map(
            move |(byte, child)| -> Box<dyn Iterator<Item = (Vec<u8>, &Self::Value)> + '_> {
                let value = if let Some(value) = child.get_value() {
                    vec![(vec![*byte], value)]
                } else {
                    vec![]
                };
                Box::new(
                    value
                        .into_iter()
                        .chain(child.find_continuations())
                        .map(|(cont, val)| {
                            (
                                vec![*byte].into_iter().chain(cont.into_iter()).collect(),
                                val,
                            )
                        }),
                )
            },
        ))
    }
}

pub struct Continuations<I> {
    inner: I,
}

impl<I> Iterator for Continuations<I>
where
    I: Iterator,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl<N, V> PrefixTreeSearch<V> for N
where
    N: Sized + PrefixTreeNode<Value = V> + Default,
{
    fn size(&self) -> usize {
        self.get_children()
            .map(|(_, child)| child.size())
            .sum::<usize>()
            + 1
    }

    #[inline]
    fn insert(&mut self, key: &str, value: V) {
        let mut node = self;
        let key = normalize(key, Normalization::NFKC, true);
        for idx in key.as_bytes() {
            if node.get_child(idx).is_none() {
                node.set_child(idx, Self::default());
            }
            node = node.get_child_mut(idx).unwrap();
        }
        node.set_value(value);
    }

    #[inline]
    fn contains(&self, prefix: &[u8]) -> bool {
        self.find(prefix).is_some()
    }

    fn contains_continuations(&self, prefix: &[u8], continuations: &[&[u8]]) -> Vec<bool> {
        let Some(node) = self.find(prefix) else {
            return vec![false; continuations.len()];
        };
        continuations
            .iter()
            .map(|cont| node.contains(cont))
            .collect()
    }

    #[inline]
    fn get(&self, prefix: &[u8]) -> Option<&V> {
        match self.find(prefix) {
            Some(node) => node.get_value(),
            None => None,
        }
    }

    fn get_continuations(&self, prefix: &[u8]) -> Box<dyn Iterator<Item = (String, &V)> + '_> {
        let Some(node) = self.find(prefix) else {
            return Box::new(std::iter::empty());
        };
        let prefix = prefix.to_vec();
        Box::new(node.find_continuations().map(move |(cont, val)| {
            let full_cont: Vec<_> = prefix.iter().cloned().chain(cont).collect();
            let key = String::from_utf8_lossy(&full_cont).to_string();
            (key, val)
        }))
    }
}

pub struct Node<V> {
    pub value: Option<V>,
    children: BTreeMap<u8, Box<Node<V>>>,
}

impl<V> Default for Node<V> {
    fn default() -> Self {
        Self {
            value: None,
            children: BTreeMap::new(),
        }
    }
}

impl<V> PrefixTreeNode for Node<V> {
    type Value = V;

    #[inline]
    fn get_child(&self, key: &u8) -> Option<&Self> {
        self.children.get(key).map(|node| node.as_ref())
    }

    fn get_child_mut(&mut self, key: &u8) -> Option<&mut Self> {
        self.children.get_mut(key).map(|node| node.as_mut())
    }

    #[inline]
    fn get_value(&self) -> Option<&Self::Value> {
        self.value.as_ref()
    }

    fn set_value(&mut self, value: Self::Value) {
        self.value = Some(value);
    }

    fn set_child(&mut self, key: &u8, value: Self) {
        self.children.insert(*key, Box::new(value));
    }

    fn get_children(&self) -> Box<dyn Iterator<Item = (&u8, &Self)> + '_> {
        Box::new(self.children.iter().map(|(k, v)| (k, v.as_ref())))
    }
}

impl<S, V> FromIterator<(S, V)> for Node<V>
where
    S: AsRef<str>,
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = (S, V)>,
    {
        let mut tree = Self::default();
        for (key, value) in iter {
            tree.insert(key.as_ref(), value);
        }
        tree
    }
}
