use crate::utils::{py_invalid_type_error, run_length_encode};
use pyo3::prelude::*;
use std::fmt::{Display, Formatter};
use std::str::Chars;
use unicode_normalization::UnicodeNormalization;
use unicode_segmentation::UnicodeSegmentation;

#[derive(Debug, Clone)]
pub struct CharString<'a> {
    pub str: &'a str,
    pub rle_cluster_lengths: Vec<(usize, usize)>,
    len: usize,
}

// shorthand for char string for in crate usage
pub(crate) type CS<'a> = CharString<'a>;

// Rust strings vs. Python
// -----------------------
// Example string "नमस्ते" from the Rust documentation
// "नमस्ते".len() -> 18; num bytes (equal to len("नमस्ते".encode()) in Python)
// "नमस्ते".chars().count() -> 6; num unicode code units (equal to len("नमस्ते") in Python)
// CharString::from("नमस्ते").len() -> 4; num grapheme clusters, closest to what
// humans consider to be characters (in Python available via third party libraries)

impl<'a> CharString<'a> {
    pub fn new(str: &'a str, use_graphemes: bool) -> CharString {
        let cluster_lengths: Vec<usize> = if use_graphemes {
            str.graphemes(true).map(|s| s.len()).collect()
        } else {
            str.chars().map(|c| c.len_utf8()).collect()
        };
        let rle_cluster_lengths = run_length_encode(&cluster_lengths);
        CharString {
            str,
            rle_cluster_lengths,
            len: cluster_lengths.len(),
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub(crate) fn byte_start_end(&self, n: usize) -> (usize, usize) {
        let mut start = 0;
        let mut total_count = 0;
        for (num_bytes, count) in &self.rle_cluster_lengths {
            if n < total_count + *count {
                start += num_bytes * (n - total_count);
                let end = start + num_bytes;
                return (start, end);
            }
            start += count * num_bytes;
            total_count += count;
        }
        panic!("should not happen")
    }

    #[inline]
    pub(crate) fn char_byte_len(&self, n: usize) -> usize {
        let (start, end) = self.byte_start_end(n);
        end - start
    }

    #[inline]
    pub(crate) fn char_range_to_byte_range(&self, start: usize, end: usize) -> (usize, usize) {
        assert!(start < end && end <= self.len());
        let (start_byte, mut end_byte) = self.byte_start_end(start);
        if start < end - 1 {
            let (_, new_end_byte) = self.byte_start_end(end - 1);
            end_byte = new_end_byte;
        }
        (start_byte, end_byte)
    }

    pub fn get(&self, n: usize) -> &str {
        assert!(n < self.len());
        let (start, end) = self.byte_start_end(n);
        &self.str[start..end]
    }

    pub fn get_char(&self, n: usize) -> Character {
        Character { str: self.get(n) }
    }

    pub fn sub(&self, start: usize, end: usize) -> &'a str {
        assert!(
            start <= end && end <= self.len(),
            "start: {start}, end: {end}, len: {}",
            self.len()
        );
        if self.len() == 0 || start == end {
            return "";
        }
        let (start, end) = self.char_range_to_byte_range(start, end);
        &self.str[start..end]
    }

    pub fn chars(&self) -> Characters {
        Characters {
            char_str: self,
            idx: 0,
        }
    }
}

impl Display for CharString<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.str,)
    }
}

pub struct Characters<'a> {
    char_str: &'a CharString<'a>,
    idx: usize,
}

impl<'a> Iterator for Characters<'a> {
    type Item = Character<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.char_str.len {
            return None;
        }
        let (start, end) = self.char_str.byte_start_end(self.idx);
        let char = Character {
            str: &self.char_str.str[start..end],
        };
        self.idx += 1;
        Some(char)
    }
}

#[derive(Debug)]
pub struct Character<'a> {
    pub str: &'a str,
}

impl<'a> Character<'a> {
    pub fn byte_len(&self) -> usize {
        self.str.len()
    }

    pub fn is_ascii(&self) -> bool {
        self.byte_len() == 1
    }

    pub fn is_whitespace(&self) -> bool {
        self.str.chars().all(char::is_whitespace)
    }

    pub fn code_point_len(&self) -> usize {
        self.str.chars().count()
    }

    pub fn code_points(&self) -> Chars {
        self.str.chars()
    }
}

impl PartialEq<Self> for Character<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.str == other.str
    }
}

impl Display for Character<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.str)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Normalization {
    NFC,
    NFD,
    NFKC,
    NFKD,
}

impl<'a> FromPyObject<'a> for Normalization {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let s: String = ob.extract()?;
        let norm = match s.as_str() {
            "nfc" | "NFC" => Normalization::NFC,
            "nfd" | "NFD" => Normalization::NFD,
            "nfkc" | "NFKC" => Normalization::NFKC,
            "nfkd" | "NFKD" => Normalization::NFKD,
            k => return Err(py_invalid_type_error(k, "normalization")),
        };
        Ok(norm)
    }
}

impl IntoPy<PyObject> for Normalization {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            Normalization::NFC => "nfc",
            Normalization::NFD => "nfd",
            Normalization::NFKC => "nfkc",
            Normalization::NFKD => "nfkd",
        }
        .into_py(py)
    }
}

#[pyfunction(signature = (s, normalization = Normalization::NFKC, use_graphemes = true))]
#[inline]
pub fn normalize(s: &str, normalization: Normalization, use_graphemes: bool) -> String {
    if use_graphemes {
        // split in grapheme clusters first, then normalize
        // every cluster individually, and concatenate the result
        let cs = CharString::new(s, use_graphemes);
        cs.chars()
            .map(|c| normalize(c.str, normalization, false))
            .collect()
    } else {
        let chars = s.chars();
        match normalization {
            Normalization::NFC => chars.nfc().collect(),
            Normalization::NFD => chars.nfd().collect(),
            Normalization::NFKC => chars.nfkc().collect(),
            Normalization::NFKD => chars.nfkd().collect(),
        }
    }
}

/// A submodule containing functionality for handling unicode.
pub(super) fn add_submodule(py: Python<'_>, parent_module: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "unicode")?;
    m.add_function(wrap_pyfunction!(normalize, m)?)?;
    parent_module.add_submodule(m)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::unicode::CS;

    #[test]
    fn test_char_string() {
        // test with empty string
        let s = CS::new("", false);
        assert_eq!(s.str.len(), 0);
        assert_eq!(s.len(), 0);
        assert_eq!(s.sub(0, s.len()), "");
        assert_eq!(s.rle_cluster_lengths, vec![]);
        // test with ascii string
        let s = CS::new("this is a test", false);
        assert_eq!(s.str.len(), 14);
        assert_eq!(s.len(), 14);
        assert_eq!(s.sub(10, 14), "test");
        assert_eq!(s.sub(10, 10), "");
        assert_eq!(s.rle_cluster_lengths, vec![(1, 14)]);
        // test with non ascii string
        let s = CS::new("this is a täst", false);
        assert_eq!(s.str.len(), 15);
        assert_eq!(s.len(), 14);
        assert_eq!(s.sub(10, 14), "täst");
        assert_eq!(s.rle_cluster_lengths, vec![(1, 11), (2, 1), (1, 2)]);
        assert_eq!(s.get(11), "ä");

        // test with string that has more than one code point per character:
        // the string "नमस्ते" should have 18 utf8 bytes, 6 code points, and 4 grapheme
        // clusters (characters)

        // first test with regular char string, which should behave unexpected
        let s = CS::new("नमस्ते", false);
        assert_eq!(s.str.len(), 18);
        assert_ne!(s.len(), 4);
        assert_ne!(s.get(2), "स्");
        assert_ne!(s.sub(2, 4), "स्ते");

        // now test with grapheme based char string, which should behave as expected
        let s = CS::new("नमस्ते", true);
        assert_eq!(s.str.len(), 18);
        assert_eq!(s.len(), 4);
        assert_eq!(s.get(2), "स्");
        assert_eq!(s.sub(2, 4), "स्ते");
    }
}
