use itertools::Itertools;
use regex::{escape, Regex};
use pyo3::prelude::*;
use crate::text::clean;
use crate::unicode::{Character, CS};

pub fn remove(s: &str) -> String {
    s.split_whitespace().join("")
}

#[pyfunction]
#[pyo3(name = "remove")]
fn remove_py(
    s: &str
) -> PyResult<String> {
    Ok(remove(s))
}

pub fn full(s: &str, use_graphemes: bool) -> String {
    s
        .split_whitespace()
        .map(|w| CS::new(w, use_graphemes).chars().join(" "))
        .join(" ")
}

#[pyfunction]
#[pyo3(name = "full")]
fn full_py(
    s: &str,
    use_graphemes: bool,
) -> PyResult<String> {
    Ok(full(s, use_graphemes))
}

pub fn operations(from: &str, to: &str, use_graphemes: bool) -> Vec<u8> {
    assert_eq!(
        remove(from),
        remove(to),
        "from and to should only differ in whitespaces"
    );
    let from_cs = CS::new(from, use_graphemes);
    let to_cs = CS::new(to, use_graphemes);
    let from_chars: Vec<Character> = from_cs.chars().collect();
    let to_chars: Vec<Character> = to_cs.chars().collect();
    let mut operations = vec![];
    operations.reserve(from_chars.len().max(to_chars.len()));
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
            operations.push(0);
            to_ptr += 1;
        } else if to_char.is_some() && to_char.unwrap().is_whitespace() {
            operations.push(1);
            to_ptr += 2;
        } else if from_char.is_whitespace() {
            operations.push(2);
        } else {
            panic!("should not happen");
        }
        from_ptr += 1;
    }
    operations
}

#[pyfunction]
#[pyo3(name = "operations")]
fn operations_py(
    from: &str,
    to: &str,
    use_graphemes: bool,
) -> PyResult<Vec<u8>> {
    Ok(operations(from, to, use_graphemes))
}

pub fn repair(s: &str, operations: &[u8], use_graphemes: bool) -> String {
    let cs = CS::new(s, use_graphemes);
    let chars: Vec<Character> = cs.chars().collect();
    assert_eq!(
        chars.len(),
        operations.len(),
        "expected one operation for every character, but got {} operations and \
        {} characters",
        operations.len(),
        chars.len()
    );

    let mut new_chars = vec![];
    new_chars.reserve(operations.len());
    for (idx, (char, op)) in chars
        .iter()
        .zip(operations.iter())
        .enumerate() {
        assert!(*op <= 2, "operation should be either 0, 1, or 2, but got {}", op);
        if *op == 1
            && !char.is_whitespace()
            && (idx == 0 || !chars[idx - 1].is_whitespace()) {
            new_chars.push(" ");
            new_chars.push(char.str);
        } else if *op == 2 && char.is_whitespace() {
            continue;
        } else {
            new_chars.push(char.str);
        }
    }
    clean(new_chars.iter().join("").as_str())
}

#[pyfunction]
#[pyo3(name = "repair")]
fn repair_py(
    s: &str,
    operations: Vec<u8>,
    use_graphemes: bool,
) -> PyResult<String> {
    Ok(repair(s, &operations, use_graphemes))
}

pub fn find_substring_ignoring_whitespace(
    s: &str,
    substring: &str,
) -> Option<(usize, usize)> {
    let substring =
        substring
            .chars()
            .filter(|c| !c.is_whitespace())
            .map(|c| escape(c.to_string().as_str()))
            .join(r"\s*");
    let re = Regex::new(substring.as_str())
        .expect("invalid regex, should not happen");
    if let Some(pattern_match) = re.find(s) {
        Some((pattern_match.start(), pattern_match.end()))
    } else {
        None
    }
}

#[pyfunction]
#[pyo3(name = "find_substring_ignoring_whitespace")]
fn find_substring_ignoring_whitespace_py(
    s: &str,
    substring: &str,
) -> PyResult<Option<(usize, usize)>> {
    Ok(find_substring_ignoring_whitespace(s, substring))
}

pub(super) fn add_submodule(py: Python<'_>, parent_module: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "whitespace")?;
    m.add_function(wrap_pyfunction!(find_substring_ignoring_whitespace_py, m)?)?;
    m.add_function(wrap_pyfunction!(repair_py, m)?)?;
    m.add_function(wrap_pyfunction!(operations_py, m)?)?;
    m.add_function(wrap_pyfunction!(full_py, m)?)?;
    m.add_function(wrap_pyfunction!(remove_py, m)?)?;
    parent_module.add_submodule(m)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::whitespace::{find_substring_ignoring_whitespace, full, operations, remove, repair};

    #[test]
    fn test_remove() {
        assert_eq!(remove(" t   h is is \n\t a tes    t "), "thisisatest");
        assert_eq!(remove(""), "");
    }

    #[test]
    fn test_full() {
        assert_eq!(full(" t   h is is \n\t a tes    t ", true), "t h i s i s a t e s t");
        assert_eq!(full("", true), "");
    }

    #[test]
    fn test_operations() {
        let from = " t  h isis a test  ";
        let to = "this is a test";
        assert_eq!(operations(from, from, true), vec![0; from.chars().count()]);
        assert_eq!(operations(to, to, true), vec![0; to.chars().count()]);
        assert_eq!(
            operations(from, to, true),
            vec![2, 0, 2, 2, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2]
        );
    }

    #[test]
    fn test_repair() {
        let from = " t h isis a test  ";
        let to = "this is a test";
        assert_eq!(
            repair(from, &operations(from, to, true), true),
            to
        );
        assert_eq!(
            repair(to, &operations(to, from, true), true),
            "t h isis a test"
        );
        assert_eq!(
            repair("    ", &vec![2, 2, 2, 2], true),
            ""
        );
        assert_eq!(
            repair("  t  ", &vec![0, 2, 0, 0, 1], true),
            "t"
        );
        assert_eq!(
            repair("", &vec![], true),
            ""
        );
    }

    #[test]
    fn test_find_substring_ignoring_whitespace() {
        let s = "this is a test sentence";
        let sub = "  a te s\n t";
        let result = find_substring_ignoring_whitespace(s, sub);
        assert!(result.is_some());
        let (start, end) = result.unwrap();
        assert_eq!(start, 8);
        assert_eq!(end, 14);
        assert_eq!(&s[start..end], "a test");
        let result = find_substring_ignoring_whitespace(s, "a täst");
        assert!(result.is_none());
        let s = "this is \" a \\w+ test \" sentence";
        let sub = "\"a \\w+test\"";
        let result = find_substring_ignoring_whitespace(s, sub);
        assert!(result.is_some());
    }
}
