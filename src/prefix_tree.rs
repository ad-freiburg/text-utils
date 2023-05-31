use std::collections::BTreeMap;

use pyo3::prelude::*;

pub trait PrefixTreeNode {
    type Value;

    fn get_value(&self) -> Option<&Self::Value>;

    fn set_value(&mut self, value: Self::Value);

    fn get_child(&self, key: &u8) -> Option<&Self>;

    fn set_child(&mut self, key: &u8, value: Self);

    fn get_child_mut(&mut self, key: &u8) -> Option<&mut Self>;
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
    fn find(&self, key: impl AsRef<str>) -> Option<&Self> {
        let mut node = self;
        for idx in key.as_ref().as_bytes() {
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

    fn contains_prefix(&self, prefix: impl AsRef<str>) -> bool {
        self.find(prefix).is_some()
    }

    fn get(&self, key: impl AsRef<str>) -> Option<&Self::Value> {
        match self.find(key) {
            Some(node) => node.get_value(),
            None => None,
        }
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
#[pyo3(name = "Node")]
#[derive(Default)]
pub struct PyNode {
    #[pyo3(get)]
    value: Option<PyObject>,
    children: BTreeMap<u8, Box<PyNode>>,
}

impl PrefixTreeNode for PyNode {
    type Value = PyObject;

    #[inline]
    fn get_value(&self) -> Option<&Self::Value> {
        self.value.as_ref()
    }

    fn set_value(&mut self, value: Self::Value) {
        self.value = Some(value);
    }

    #[inline]
    fn get_child(&self, key: &u8) -> Option<&Self> {
        self.children.get(key).map(|node| node.as_ref())
    }

    fn set_child(&mut self, key: &u8, value: Self) {
        self.children.insert(*key, Box::new(value));
    }

    fn get_child_mut(&mut self, key: &u8) -> Option<&mut Self> {
        self.children.get_mut(key).map(|node| node.as_mut())
    }
}

/// A submodule containing an implementation of a prefix tree
pub(super) fn add_submodule(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "prefix_tree")?;
    m.add_class::<PyNode>()?;
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
