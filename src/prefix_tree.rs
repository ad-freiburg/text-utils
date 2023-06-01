use std::collections::BTreeMap;

use pyo3::prelude::*;

pub trait PrefixTreeNode {
    type Value;

    fn get_value(&self) -> Option<&Self::Value>;

    fn set_value(&mut self, value: Self::Value);

    fn get_child(&self, key: &u8) -> Option<&Self>;

    fn get_children(&self) -> Box<dyn Iterator<Item = (&u8, &Self)> + '_>;

    fn set_child(&mut self, key: &u8, value: Self);

    fn get_child_mut(&mut self, key: &u8) -> Option<&mut Self>;
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

pub trait PrefixTreeSearch
where
    Self: Sized + PrefixTreeNode + Default,
{
    #[inline]
    fn is_terminal(&self) -> bool {
        self.get_value().is_some()
    }

    #[inline]
    fn insert(&mut self, key: impl AsRef<str>, value: Self::Value) {
        let mut node = self;
        for idx in key.as_ref().as_bytes() {
            if node.get_child(idx).is_none() {
                node.set_child(idx, Self::default());
            }
            node = node.get_child_mut(idx).unwrap();
        }
        node.set_value(value);
    }

    #[inline]
    fn find(&self, prefix: impl AsRef<str>) -> Option<&Self> {
        let mut node = self;
        for idx in prefix.as_ref().as_bytes() {
            if let Some(child) = node.get_child(idx) {
                node = child;
            } else {
                return None;
            }
        }
        Some(node)
    }

    fn contains(&self, key: impl AsRef<str>) -> bool {
        match self.find(key) {
            Some(node) => node.is_terminal(),
            None => false,
        }
    }

    #[inline]
    fn contains_prefix(&self, prefix: impl AsRef<str>) -> bool {
        self.find(prefix).is_some()
    }

    fn contains_continuations(
        &self,
        key: impl AsRef<str>,
        continuations: &[impl AsRef<str>],
    ) -> Vec<bool> {
        let Some(node) = self.find(key) else {
            return vec![false; continuations.len()];
        };
        continuations
            .iter()
            .map(|cont| node.contains_prefix(cont))
            .collect()
    }

    #[inline]
    fn get(&self, key: impl AsRef<str>) -> Option<&Self::Value> {
        match self.find(key) {
            Some(node) => node.get_value(),
            None => None,
        }
    }

    fn get_continuations(
        &self,
        key: impl AsRef<str>,
        continuations: &[impl AsRef<str>],
    ) -> Vec<Option<&Self::Value>> {
        let Some(node) = self.find(key) else {
            return vec![None; continuations.len()];
        };
        continuations.iter().map(|cont| node.get(cont)).collect()
    }

    fn find_continuations_with(
        &self,
        prefix: Vec<u8>,
    ) -> Box<dyn Iterator<Item = (Vec<u8>, &Self::Value)> + '_> {
        Box::new(self.get_children().flat_map(
            move |(byte, child)| -> Box<dyn Iterator<Item = (Vec<u8>, &Self::Value)> + '_> {
                let mut pfx = prefix.clone();
                pfx.push(*byte);
                if let Some(value) = child.get_value() {
                    Box::new(vec![(pfx, value)].into_iter())
                } else {
                    Box::new(child.find_continuations_with(pfx))
                }
            },
        ))
    }

    fn find_continuations(&self) -> Box<dyn Iterator<Item = (Vec<u8>, &Self::Value)> + '_> {
        self.find_continuations_with(vec![])
    }

    fn build_from_iter<S: AsRef<str>>(iter: impl IntoIterator<Item = (S, Self::Value)>) -> Self {
        let mut tree = Self::default();
        for (key, value) in iter {
            tree.insert(key.as_ref(), value);
        }
        tree
    }
}

impl<N> PrefixTreeSearch for N where N: Sized + PrefixTreeNode + Default {}

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
        Self::build_from_iter(iter.into_iter())
    }
}

#[pyclass]
#[pyo3(name = "Tree")]
pub struct PyTree {
    root: Node<PyObject>,
}

#[pymethods]
impl PyTree {
    #[new]
    fn new() -> Self {
        Self {
            root: Node::default(),
        }
    }

    fn insert(&mut self, key: &str, value: PyObject) {
        self.root.insert(key, value);
    }

    fn contains(&self, key: &str) -> bool {
        self.root.contains(key)
    }

    fn contains_prefix(&self, prefix: &str) -> bool {
        self.root.contains_prefix(prefix)
    }

    fn get(&self, key: &str) -> Option<&PyObject> {
        self.root.get(key)
    }

    fn contains_continuations(&self, prefix: &str, continuations: Vec<&str>) -> Vec<bool> {
        self.root.contains_continuations(prefix, &continuations)
    }

    fn get_continuations(&self, prefix: &str, continuations: Vec<&str>) -> Vec<Option<&PyObject>> {
        self.root.get_continuations(prefix, &continuations)
    }

    fn find_continuations(&self, prefix: &str) -> Vec<(String, &PyObject)> {
        let Some(node) = self.root.find(prefix) else {
            return vec![];
        };
        node.find_continuations_with(prefix.as_bytes().to_vec())
            .map(|(k, v)| (String::from_utf8_lossy(&k).to_string(), v))
            .collect()
    }
}

/// A submodule containing an implementation of a prefix tree
pub(super) fn add_submodule(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "prefix_tree")?;
    m.add_class::<PyTree>()?;
    parent_module.add_submodule(m)?;

    Ok(())
}

#[cfg(test)]
pub mod tests {
    use crate::prefix_tree::PrefixTreeSearch;

    #[test]
    fn test_prefix_tree() {
        let mut tree = super::Node::default();
        tree.insert("hello", 1);
        assert!(tree.contains("hello"));
        assert!(!tree.contains("hell"));
        assert!(tree.contains_prefix("hell"));
        assert!(!tree.contains("helloo"));
        assert!(tree.contains_prefix(""));
        assert!(tree.get("hell").is_none());
        assert_eq!(tree.get("hello"), Some(&1));
        tree.insert("hello", 2);
        assert_eq!(tree.get("hello"), Some(&2));
        tree = [("hello", 1), ("hell", 2)].into_iter().collect();
        assert_eq!(tree.get("hello"), Some(&1));
        assert_eq!(tree.get("hell"), Some(&2));
        // get subtrees via find
        let subtree = tree.find("he").unwrap();
        assert_eq!(subtree.value, None);
        let subtree = subtree.find("ll").unwrap();
        assert_eq!(subtree.value, Some(2));
        let subtree = subtree.find("o").unwrap();
        assert_eq!(subtree.value, Some(1));
    }
}
