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

    #[inline]
    fn contains_prefix(&self, prefix: impl AsRef<str>) -> bool {
        self.find(prefix).is_some()
    }

    fn contains_continuations(
        &self,
        prefix: impl AsRef<str>,
        continuations: &[impl AsRef<str>],
    ) -> Vec<bool> {
        let Some(node) = self.find(prefix) else {
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
}

#[pyclass]
#[pyo3(name = "Node")]
#[derive(Default)]
pub struct PyNode {
    #[pyo3(get)]
    value: Option<PyObject>,
    children: BTreeMap<u8, Py<PyNode>>,
}

impl PyNode {
    #[inline]
    fn get_child(&self, py: Python<'_>, key: &u8) -> Option<Py<PyNode>> {
        self.children.get(key).map(move |node| node.clone_ref(py))
    }
}

#[pymethods]
impl PyNode {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    fn insert(&mut self, py: Python<'_>, key: &str, value: PyObject) -> anyhow::Result<()> {
        let bytes = key.as_bytes();
        if bytes.is_empty() {
            return Err(anyhow::anyhow!("key is empty"));
        }
        let mut node = match self.get_child(py, &bytes[0]) {
            Some(node) => node,
            None => {
                let node = Py::new(py, PyNode::new())?;
                self.children.insert(bytes[0], node.clone_ref(py));
                node
            }
        };
        for byte in &bytes[1..] {
            let next = node.as_ref(py).borrow().get_child(py, byte);
            if let Some(next) = next {
                node = next;
                continue;
            };
            let next = Py::new(py, PyNode::new())?;
            node.as_ref(py)
                .borrow_mut()
                .children
                .insert(*byte, next.clone_ref(py));
            node = next;
        }
        node.as_ref(py).borrow_mut().value = Some(value);
        Ok(())
    }

    fn is_terminal(&self) -> bool {
        self.value.is_some()
    }

    fn find(&self, py: Python<'_>, key: &str) -> Option<Py<Self>> {
        let bytes = key.as_bytes();
        if bytes.is_empty() {
            return None;
        }
        let mut node: Py<Self> = match self.get_child(py, &bytes[0]) {
            Some(child) => child,
            None => return None,
        };
        for byte in &bytes[1..] {
            let next = match node.as_ref(py).borrow().get_child(py, byte) {
                Some(child) => child,
                None => return None,
            };
            node = next;
        }
        Some(node)
    }

    fn contains_prefix(&self, py: Python<'_>, key: &str) -> bool {
        self.find(py, key).is_some()
    }

    fn contains(&self, py: Python<'_>, key: &str) -> bool {
        match self.find(py, key) {
            Some(node) => {
                let node_ref = node.as_ref(py);
                node_ref.borrow().is_terminal()
            }
            None => false,
        }
    }

    fn get(&self, py: Python<'_>, key: &str) -> Option<PyObject> {
        match self.find(py, key) {
            None => None,
            Some(node) => node
                .as_ref(py)
                .borrow()
                .value
                .as_ref()
                .map(|value| value.clone_ref(py)),
        }
    }
}

/// A submodule containing an implementation of a prefix tree
pub(super) fn add_submodule(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "prefix_tree")?;
    m.add_class::<PyNode>()?;
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
