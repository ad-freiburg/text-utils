use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use pyo3::prelude::*;
use rayon::prelude::*;

use crate::{prefix_tree::Node, prefix_vec::PrefixVec, utils::SerializeMsgPack};

pub trait PrefixTreeSearch<V> {
    fn size(&self) -> usize;

    fn insert(&mut self, key: &[u8], value: V);

    fn get(&self, prefix: &[u8]) -> Option<&V>;

    fn get_mut(&mut self, prefix: &[u8]) -> Option<&mut V>;

    fn contains(&self, prefix: &[u8]) -> bool;

    fn get_continuations(&self, prefix: &[u8]) -> Box<dyn Iterator<Item = (Vec<u8>, &V)> + '_>;
}

#[pyclass]
#[pyo3(name = "Tree")]
pub struct PyPrefixTree {
    inner: Node<String>,
    continuations: Option<Vec<Vec<u8>>>,
}

#[pyclass]
#[pyo3(name = "Vec")]
pub struct PyPrefixVec {
    inner: PrefixVec<String>,
}

#[pymethods]
impl PyPrefixTree {
    #[new]
    fn new() -> Self {
        Self {
            inner: Node::default(),
            continuations: None,
        }
    }

    fn __len__(&self) -> usize {
        self.inner.size()
    }

    #[staticmethod]
    fn load(path: &str) -> anyhow::Result<Self> {
        let inner = Node::load(path)?;
        Ok(Self {
            inner,
            continuations: None,
        })
    }

    fn save(&self, path: &str) -> anyhow::Result<()> {
        self.inner.save(path)?;
        Ok(())
    }

    #[staticmethod]
    fn from_file(path: &str) -> anyhow::Result<Self> {
        let file = File::open(path)?;
        let inner = BufReader::new(file)
            .lines()
            .filter_map(|line| match line {
                Err(_) => None,
                Ok(s) => {
                    let splits: Vec<_> = s.split('\t').collect();
                    assert!(splits.len() >= 3, "invalid line: {}", s);
                    let value = splits[0].trim();
                    Some(
                        splits[2..]
                            .iter()
                            .map(|&s| (s.trim().as_bytes().to_vec(), value.to_string()))
                            .collect::<Vec<_>>(),
                    )
                }
            })
            .flatten()
            .collect();
        Ok(Self {
            inner,
            continuations: None,
        })
    }

    fn insert(&mut self, key: &[u8], value: String) {
        self.inner.insert(key, value);
    }

    fn contains(&self, prefix: Vec<u8>) -> bool {
        self.inner.contains(&prefix)
    }

    fn get(&self, key: Vec<u8>) -> Option<&str> {
        self.inner.get(&key).map(|s| s.as_ref())
    }

    fn batch_get(&self, keys: Vec<Vec<u8>>) -> Vec<Option<&str>> {
        keys.into_iter()
            .map(|key| self.inner.get(&key).map(|s| s.as_ref()))
            .collect()
    }

    fn set_continuations(&mut self, continuations: Vec<Vec<u8>>) {
        self.continuations = Some(continuations);
    }

    fn get_continuations(&self, prefix: Vec<u8>) -> Vec<(Vec<u8>, &str)> {
        self.inner
            .get_continuations(&prefix)
            .map(|(s, v)| (s.to_vec(), v.as_ref()))
            .collect()
    }

    fn batch_get_continuations(&self, prefixes: Vec<Vec<u8>>) -> Vec<Vec<(Vec<u8>, &str)>> {
        prefixes
            .into_par_iter()
            .map(|prefix| {
                self.inner
                    .get_continuations(&prefix)
                    .map(|(s, v)| (s.to_vec(), v.as_ref()))
                    .collect()
            })
            .collect()
    }
}

#[pymethods]
impl PyPrefixVec {
    #[new]
    fn new() -> Self {
        Self {
            inner: PrefixVec::default(),
        }
    }

    fn __len__(&self) -> usize {
        self.inner.size()
    }

    #[staticmethod]
    fn load(path: &str) -> anyhow::Result<Self> {
        let inner = PrefixVec::load(path)?;
        Ok(Self { inner })
    }

    fn save(&mut self, path: &str) -> anyhow::Result<()> {
        let cont = self.inner.cont.take();
        self.inner.save(path)?;
        self.inner.cont = cont;
        Ok(())
    }

    #[staticmethod]
    fn from_file(path: &str) -> anyhow::Result<Self> {
        let file = File::open(path)?;
        let inner = BufReader::new(file)
            .lines()
            .filter_map(|line| match line {
                Err(_) => None,
                Ok(s) => {
                    let splits: Vec<_> = s.split('\t').collect();
                    assert!(splits.len() >= 3, "invalid line: {}", s);
                    let value = splits[0].trim();
                    Some(
                        splits[2..]
                            .iter()
                            .map(|&s| (s.trim().as_bytes().to_vec(), value.to_string()))
                            .collect::<Vec<_>>(),
                    )
                }
            })
            .flatten()
            .collect();
        Ok(Self { inner })
    }

    fn insert(&mut self, key: Vec<u8>, value: String) {
        self.inner.insert(&key, value);
    }

    fn contains(&self, prefix: Vec<u8>) -> bool {
        self.inner.contains(&prefix)
    }

    fn batch_contains(&self, prefixes: Vec<Vec<u8>>) -> Vec<bool> {
        prefixes
            .into_iter()
            .map(|prefix| self.inner.contains(&prefix))
            .collect()
    }

    fn get(&self, key: Vec<u8>) -> Option<&str> {
        self.inner.get(&key).map(|s| s.as_ref())
    }

    fn batch_get(&self, keys: Vec<Vec<u8>>) -> Vec<Option<&str>> {
        keys.into_iter()
            .map(|key| self.inner.get(&key).map(|s| s.as_ref()))
            .collect()
    }

    fn set_continuations(&mut self, continuations: Vec<Vec<u8>>) {
        self.inner.set_continuations(continuations)
    }

    fn contains_continuations(&self, prefix: Vec<u8>) -> anyhow::Result<Vec<bool>> {
        self.inner.contains_continuations(&prefix)
    }

    fn batch_contains_continuations(
        &self,
        prefixes: Vec<Vec<u8>>,
    ) -> anyhow::Result<Vec<Vec<bool>>> {
        prefixes
            .into_par_iter()
            .map(|prefix| self.inner.contains_continuations(&prefix))
            .collect::<anyhow::Result<_>>()
    }

    fn get_continuations(&self, prefix: Vec<u8>) -> Vec<(Vec<u8>, &str)> {
        self.inner
            .get_continuations(&prefix)
            .map(|(s, v)| (s.to_vec(), v.as_ref()))
            .collect()
    }

    fn batch_get_continuations(&self, prefixes: Vec<Vec<u8>>) -> Vec<Vec<(Vec<u8>, &str)>> {
        prefixes
            .into_par_iter()
            .map(|prefix| {
                self.inner
                    .get_continuations(&prefix)
                    .map(|(s, v)| (s.to_vec(), v.as_ref()))
                    .collect()
            })
            .collect()
    }

    fn at(&self, idx: usize) -> Option<(Vec<u8>, &str)> {
        self.inner
            .data
            .get(idx)
            .map(|(s, v)| (s.to_vec(), v.as_ref()))
    }
}

/// A submodule containing two implementations of a prefix tree
pub(super) fn add_submodule(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "prefix")?;
    m.add_class::<PyPrefixTree>()?;
    m.add_class::<PyPrefixVec>()?;
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
            tree.insert(b"hello", 1);
            assert!(tree.contains(b"hello"));
            assert!(tree.contains(b"hell"));
            assert!(!tree.contains(b"helloo"));
            assert!(tree.get(b"hell").is_none());
            assert_eq!(tree.get(b"hello"), Some(&1));
            tree.insert(b"hello", 2);
            assert_eq!(tree.get(b"hello"), Some(&2));
        }
    }
}
