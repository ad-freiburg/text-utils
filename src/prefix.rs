use anyhow::anyhow;
use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use pyo3::prelude::*;

use crate::{prefix_tree::Node, prefix_vec::PrefixVec, utils::SerializeMsgPack};

pub trait PrefixTreeSearch<V> {
    fn size(&self) -> usize;

    fn insert(&mut self, key: &str, value: V);

    fn get(&self, prefix: &[u8]) -> Option<&V>;

    fn contains(&self, prefix: &[u8]) -> bool;

    fn get_continuations(&self, prefix: &[u8]) -> Box<dyn Iterator<Item = (String, &V)> + '_>;

    fn contains_continuations(&self, prefix: &[u8], continuations: &[Vec<u8>]) -> Vec<bool>;
}

#[pyclass]
#[pyo3(name = "Tree")]
pub struct PyPrefixTree {
    inner: Node<usize>,
    continuations: Option<Vec<Vec<u8>>>,
}

#[pyclass]
#[pyo3(name = "Vec")]
pub struct PyPrefixVec {
    inner: PrefixVec<usize>,
    continuations: Option<Vec<Vec<u8>>>,
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
                    assert!(splits.len() == 2, "invalid line: {}", s);
                    let value: usize = splits[1]
                        .trim()
                        .parse()
                        .unwrap_or_else(|_| panic!("failed to parse {} into usize", splits[1]));
                    Some((splits[0].trim().to_string(), value))
                }
            })
            .collect();
        Ok(Self {
            inner,
            continuations: None,
        })
    }

    fn insert(&mut self, key: &str, value: usize) {
        self.inner.insert(key, value);
    }

    fn contains(&self, prefix: Vec<u8>) -> bool {
        self.inner.contains(&prefix)
    }

    fn get(&self, key: Vec<u8>) -> Option<usize> {
        self.inner.get(&key).copied()
    }

    fn set_continuations(&mut self, continuations: Vec<Vec<u8>>) {
        self.continuations = Some(continuations);
    }

    fn contains_continuations(&self, prefix: Vec<u8>) -> anyhow::Result<Vec<bool>> {
        let continuations = self
            .continuations
            .as_ref()
            .ok_or_else(|| anyhow!("continuations not set"))?;
        Ok(self.inner.contains_continuations(&prefix, continuations))
    }

    fn get_continuations(&self, prefix: Vec<u8>) -> Vec<(String, usize)> {
        self.inner
            .get_continuations(&prefix)
            .map(|(s, v)| (s, *v))
            .collect()
    }
}

#[pymethods]
impl PyPrefixVec {
    #[new]
    fn new() -> Self {
        Self {
            inner: PrefixVec::default(),
            continuations: None,
        }
    }

    #[staticmethod]
    fn load(path: &str) -> anyhow::Result<Self> {
        let inner = PrefixVec::load(path)?;
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
                    assert!(splits.len() == 2, "invalid line: {}", s);
                    let value: usize = splits[1]
                        .trim()
                        .parse()
                        .unwrap_or_else(|_| panic!("failed to parse {} into usize", splits[1]));
                    Some((splits[0].trim().to_string(), value))
                }
            })
            .collect();
        Ok(Self {
            inner,
            continuations: None,
        })
    }

    fn insert(&mut self, key: &str, value: usize) {
        self.inner.insert(key, value);
    }

    fn contains(&self, prefix: Vec<u8>) -> bool {
        self.inner.contains(&prefix)
    }

    fn get(&self, key: Vec<u8>) -> Option<usize> {
        self.inner.get(&key).copied()
    }

    fn set_continuations(&mut self, continuations: Vec<Vec<u8>>) {
        self.continuations = Some(continuations);
    }

    fn contains_continuations(&self, prefix: Vec<u8>) -> anyhow::Result<Vec<bool>> {
        let continuations = self
            .continuations
            .as_ref()
            .ok_or_else(|| anyhow!("continuations not set"))?;
        Ok(self.inner.contains_continuations(&prefix, continuations))
    }

    fn get_continuations(&self, prefix: Vec<u8>) -> Vec<(String, usize)> {
        self.inner
            .get_continuations(&prefix)
            .map(|(s, v)| (s, *v))
            .collect()
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
