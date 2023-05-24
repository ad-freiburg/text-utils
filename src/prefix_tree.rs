use std::collections::BTreeMap;

use pyo3::prelude::*;

struct Node<T> {
    value: Option<T>,
    children: BTreeMap<u8, Box<Node<T>>>,
}

impl<T> Node<T> {
    fn new(value: Option<T>) -> Self {
        Node {
            value,
            children: BTreeMap::new(),
        }
    }

    #[inline]
    fn is_terminal(&self) -> bool {
        self.value.is_some()
    }
}

pub struct PrefixTree<T> {
    root: Node<T>,
    num_nodes: usize,
    num_values: usize,
}

impl<T> Default for PrefixTree<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> PrefixTree<T> {
    pub fn new() -> Self {
        PrefixTree {
            root: Node::new(None),
            num_nodes: 0,
            num_values: 0,
        }
    }

    pub fn insert(&mut self, key: impl AsRef<str>, value: T) {
        let mut node = &mut self.root;
        for idx in key.as_ref().as_bytes() {
            if !node.children.contains_key(idx) {
                self.num_nodes += 1;
                node.children.insert(*idx, Box::new(Node::new(None)));
            }
            node = node.children.get_mut(idx).unwrap();
        }
        if node.value.is_none() {
            self.num_values += 1;
        }
        node.value = Some(value);
    }

    #[inline]
    fn find(&self, key: impl AsRef<str>) -> Option<&Node<T>> {
        let mut node = &self.root;
        for idx in key.as_ref().as_bytes() {
            if !node.children.contains_key(idx) {
                return None;
            }
            node = node.children.get(idx).unwrap();
        }
        Some(node)
    }

    pub fn contains(&self, key: impl AsRef<str>) -> bool {
        match self.find(key) {
            Some(node) => node.is_terminal(),
            None => false,
        }
    }

    pub fn contains_prefix(&self, prefix: impl AsRef<str>) -> bool {
        self.find(prefix).is_some()
    }

    pub fn get(&self, key: impl AsRef<str>) -> Option<&T> {
        match self.find(key) {
            Some(node) => node.value.as_ref(),
            None => None,
        }
    }

    fn build_from_iter<S: AsRef<str>>(iter: impl IntoIterator<Item = (S, T)>) -> Self {
        let mut tree = PrefixTree::new();
        for (key, value) in iter {
            tree.insert(key.as_ref(), value);
        }
        tree
    }
}

impl<S, T> FromIterator<(S, T)> for PrefixTree<T>
where
    S: AsRef<str>,
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = (S, T)>,
    {
        PrefixTree::build_from_iter(iter.into_iter())
    }
}

#[pyclass]
#[pyo3(name = "PrefixTree")]
pub struct PyPrefixTree {
    tree: PrefixTree<PyObject>,
}

#[pymethods]
impl PyPrefixTree {
    #[new]
    fn new() -> Self {
        PyPrefixTree {
            tree: PrefixTree::new(),
        }
    }

    fn size(&self) -> (usize, usize) {
        (self.tree.num_nodes, self.tree.num_values)
    }

    fn insert(&mut self, key: &str, value: PyObject) {
        self.tree.insert(key, value);
    }

    fn contains(&self, key: &str) -> bool {
        self.tree.contains(key)
    }

    fn contains_prefix(&self, prefix: &str) -> bool {
        self.tree.contains_prefix(prefix)
    }

    fn get(&self, key: &str) -> Option<&PyObject> {
        self.tree.get(key)
    }
}

/// A submodule containing an implementation of a prefix tree
pub(super) fn add_submodule(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "prefix_tree")?;
    m.add_class::<PyPrefixTree>()?;
    parent_module.add_submodule(m)?;

    Ok(())
}

#[cfg(test)]
pub mod tests {
    #[test]
    fn test_prefix_tree() {
        let mut tree = super::PrefixTree::new();
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
    }
}
