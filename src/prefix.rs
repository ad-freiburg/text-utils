use pyo3::prelude::*;

use crate::{prefix_tree::Node, prefix_vec::PrefixVec};

pub trait PrefixTreeSearch<V> {
    fn size(&self) -> usize;

    fn insert(&mut self, key: &str, value: V);

    fn get(&self, prefix: &[u8]) -> Option<&V>;

    fn contains(&self, prefix: &[u8]) -> bool;

    fn get_continuations(&self, prefix: &[u8]) -> Box<dyn Iterator<Item = (String, &V)> + '_>;

    fn contains_continuations(&self, prefix: &[u8], continuations: &[&[u8]]) -> Vec<bool>;
}

#[pyclass]
pub struct Tree {
    tree: Box<dyn PrefixTreeSearch<PyObject> + Send + Sync + 'static>,
}

#[pymethods]
impl Tree {
    #[new]
    #[pyo3(signature = (memory_efficient = false))]
    fn new(memory_efficient: bool) -> Self {
        Self {
            tree: if memory_efficient {
                Box::<PrefixVec<PyObject>>::default()
            } else {
                Box::<Node<PyObject>>::default()
            },
        }
    }

    fn insert(&mut self, key: &str, value: PyObject) {
        self.tree.insert(key, value);
    }

    fn contains(&self, key: &str) -> bool {
        self.tree.contains(key.as_bytes())
    }

    fn get(&self, key: &str) -> Option<&PyObject> {
        self.tree.get(key.as_bytes())
    }

    fn contains_continuations(&self, prefix: &str, continuations: Vec<&str>) -> Vec<bool> {
        self.tree.contains_continuations(
            prefix.as_bytes(),
            continuations
                .into_iter()
                .map(|c| c.as_bytes())
                .collect::<Vec<_>>()
                .as_slice(),
        )
    }

    fn get_continuations(&self, prefix: &str) -> Vec<(String, &PyObject)> {
        self.tree.get_continuations(prefix.as_bytes()).collect()
    }
}

/// A submodule containing an implementation of a prefix tree
pub(super) fn add_submodule(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "prefix")?;
    m.add_class::<Tree>()?;
    parent_module.add_submodule(m)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::{prefix::PrefixTreeSearch, prefix_tree::Node, prefix_vec::PrefixVec};

    #[test]
    fn test_prefix() {
        let trees: Vec<Box<dyn PrefixTreeSearch<i32>>> =
            vec![Box::new(Node::default()), Box::new(PrefixVec::default())];
        for mut tree in trees {
            tree.insert("hello", 1);
            assert!(tree.contains("hello".as_bytes()));
            assert!(tree.contains("hell".as_bytes()));
            assert!(!tree.contains("helloo".as_bytes()));
            assert!(tree.get("hell".as_bytes()).is_none());
            assert_eq!(tree.get("hello".as_bytes()), Some(&1));
            tree.insert("hello", 2);
            assert_eq!(tree.get("hello".as_bytes()), Some(&2));
        }
    }
}
