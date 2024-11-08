use crate::unicode::{Character, CS};
use crate::utils::py_invalid_type_error;
use anyhow::anyhow;
use itertools::Itertools;
use pyo3::prelude::*;
use regex::{escape, Regex};

#[pyfunction(signature = (s, use_graphemes = true))]
pub fn remove(s: &str, use_graphemes: bool) -> String {
    CS::new(s, use_graphemes)
        .chars()
        .filter(|c| !c.is_whitespace())
        .join("")
}

#[pyfunction(signature = (s, use_graphemes = true))]
pub fn full(s: &str, use_graphemes: bool) -> String {
    CS::new(s, use_graphemes)
        .chars()
        .filter(|c| !c.is_whitespace())
        .join(" ")
}

#[derive(Debug, Clone, Copy, Hash, Ord, PartialOrd, Eq, PartialEq)]
pub enum Operation {
    Keep,
    Insert,
    Delete,
}

impl<'a> FromPyObject<'a> for Operation {
    fn extract_bound(ob: &Bound<'a, PyAny>) -> PyResult<Self> {
        let s: PyResult<String> = ob.extract();
        let ws_op = if let Ok(s) = s {
            match s.as_str() {
                "k" | "keep" => Operation::Keep,
                "i" | "insert" => Operation::Insert,
                "d" | "delete" => Operation::Delete,
                k => return Err(py_invalid_type_error(k, "whitespace operation")),
            }
        } else {
            let s: u8 = ob.extract()?;
            match s {
                0 => Operation::Keep,
                1 => Operation::Insert,
                2 => Operation::Delete,
                k => return Err(py_invalid_type_error(k, "whitespace operation")),
            }
        };
        Ok(ws_op)
    }
}

impl IntoPy<PyObject> for Operation {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            Operation::Keep => "k",
            Operation::Insert => "i",
            Operation::Delete => "d",
        }
        .into_py(py)
    }
}

#[pyfunction(signature = (from, to, use_graphemes = true))]
pub fn operations(from: &str, to: &str, use_graphemes: bool) -> anyhow::Result<Vec<Operation>> {
    let from_cs = CS::new(from, use_graphemes);
    let to_cs = CS::new(to, use_graphemes);
    let from_chars: Vec<Character> = from_cs.chars().collect();
    let to_chars: Vec<Character> = to_cs.chars().collect();
    let mut operations = Vec::with_capacity(from_chars.len().max(to_chars.len()));
    let mut from_ptr = 0;
    let mut to_ptr = 0;
    while from_ptr < from_chars.len() {
        let from_char = &from_chars[from_ptr];
        let to_char = if to_ptr < to_chars.len() {
            Some(&to_chars[to_ptr])
        } else {
            None
        };
        if to_char.is_some() && from_char == to_char.unwrap() {
            operations.push(Operation::Keep);
            to_ptr += 1;
        } else if to_char.is_some() && to_char.unwrap().is_whitespace() {
            operations.push(Operation::Insert);
            to_ptr += 2;
        } else if from_char.is_whitespace() {
            operations.push(Operation::Delete);
        } else {
            return Err(anyhow!(
                "should not happen, most likely your inputs contain multiple \
                consecutive whitespaces, leading, or trailing whitespaces, \
                prepare them first using the clean function:\n\
                from: \"{from}\"\nto  : \"{to}\"\n\
                from_char: \"{from_char}\"\nto_char  : \"{to_char:?}\"\n"
            ));
        }
        from_ptr += 1;
    }
    Ok(operations)
}

pub fn repair(s: &str, operations: &[Operation], use_graphemes: bool) -> anyhow::Result<String> {
    let cs = CS::new(s, use_graphemes);
    let chars: Vec<Character> = cs.chars().collect();
    if chars.len() != operations.len() {
        return Err(anyhow!(
            "expected one operation for every character, but got {} operations and \
        {} characters for input\n{s}",
            operations.len(),
            chars.len()
        ));
    };

    let mut output = String::new();
    for (idx, (char, op)) in chars.iter().zip(operations.iter()).enumerate() {
        if *op == Operation::Insert
            && !char.is_whitespace()
            && (idx == 0 || !chars[idx - 1].is_whitespace())
        {
            output.push(' ');
            output.push_str(char.str);
        } else if *op == Operation::Delete && char.is_whitespace() {
            continue;
        } else {
            output.push_str(char.str);
        }
    }
    Ok(output)
}

#[pyfunction(name = "repair", signature = (s, operations, use_graphemes = true))]
fn repair_py(s: &str, operations: Vec<Operation>, use_graphemes: bool) -> anyhow::Result<String> {
    repair(s, &operations, use_graphemes)
}

pub fn find_substring_ignoring_whitespace<'a>(
    s: &'a str,
    substring: &str,
    use_graphemes: bool,
) -> Option<&'a str> {
    let cs = CS::new(substring, use_graphemes);
    let substring = r"\s*".to_string()
        + &cs
            .chars()
            .filter(|c| !c.is_whitespace())
            .map(|c| escape(c.str))
            .join(r"\s*")
        + r"\s*";
    let re = Regex::new(substring.as_str()).expect("invalid pattern, should never happen");
    re.find(s).map(|m| &s[m.start()..m.end()])
}

#[pyfunction(name = "find_substring_ignoring_whitespace", signature = (s, substring, use_graphemes = true))]
pub fn find_substring_ignoring_whitespace_py(
    s: &str,
    substring: &str,
    use_graphemes: bool,
) -> Option<String> {
    find_substring_ignoring_whitespace(s, substring, use_graphemes).map(|s| s.to_string())
}

/// A submodule containing functionality specific to handle whitespaces in text.
pub(super) fn add_submodule(py: Python<'_>, parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(py, "whitespace")?;
    m.add_function(wrap_pyfunction!(
        find_substring_ignoring_whitespace_py,
        m.clone()
    )?)?;
    m.add_function(wrap_pyfunction!(repair_py, m.clone())?)?;
    m.add_function(wrap_pyfunction!(operations, m.clone())?)?;
    m.add_function(wrap_pyfunction!(full, m.clone())?)?;
    m.add_function(wrap_pyfunction!(remove, m.clone())?)?;
    parent_module.add_submodule(&m)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::whitespace::{
        find_substring_ignoring_whitespace, full, operations, remove, repair, Operation,
    };

    #[test]
    fn test_remove() {
        assert_eq!(remove(" t   h is is \n\t a tes    t ", true), "thisisatest");
        assert_eq!(remove("", true), "");
    }

    #[test]
    fn test_full() {
        assert_eq!(
            full(" t   h is is \n\t a tes    t ", true),
            "t h i s i s a t e s t"
        );
        assert_eq!(full("", true), "");
    }

    #[test]
    fn test_operations() {
        let from = " t  h isis a test  ";
        let to = "this is a test";
        assert_eq!(
            operations(from, from, true).unwrap(),
            vec![Operation::Keep; from.chars().count()]
        );
        assert_eq!(
            operations(to, to, true).unwrap(),
            vec![Operation::Keep; to.chars().count()]
        );
        assert_eq!(
            operations(from, to, true)
                .unwrap()
                .into_iter()
                .map(|op| op as u8)
                .collect::<Vec<u8>>(),
            vec![2, 0, 2, 2, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2]
        );
    }

    #[test]
    fn test_repair() {
        let from = "t h isis a test";
        let to = "this is a test";
        assert_eq!(
            repair(from, &operations(from, to, true).unwrap(), true).unwrap(),
            to
        );
        assert_eq!(
            repair(to, &operations(to, from, true).unwrap(), true).unwrap(),
            "t h isis a test"
        );
        assert_eq!(repair("t", &vec![Operation::Delete,], true,).unwrap(), "t");
        assert_eq!(repair("", &vec![], true).unwrap(), "");
    }

    #[test]
    fn test_find_substring_ignoring_whitespace() {
        let s = "this is a test sentence";
        let sub = "  a te s\n t";
        let result = find_substring_ignoring_whitespace(s, sub, true);
        assert_eq!(result.unwrap(), " a test ");
        let result = find_substring_ignoring_whitespace(s, "a täst", true);
        assert!(result.is_none());
        let s = "this is \" a \\w+ test \" sentence";
        let sub = "\"a \\w+test\"";
        let result = find_substring_ignoring_whitespace(s, sub, true);
        assert!(result.is_some());
    }
}
