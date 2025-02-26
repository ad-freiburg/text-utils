use crate::utils::{py_invalid_type_error, run_length_decode, run_length_encode};
use pyo3::conversion::FromPyObject;
use pyo3::prelude::*;
use pyo3::types::PyString;
use regex::Regex;
use std::fmt::{Display, Formatter};
use std::str::Chars;
use unicode_normalization::UnicodeNormalization;
use unicode_segmentation::UnicodeSegmentation;

#[derive(Debug, Clone)]
pub struct CharString<'a> {
    pub str: &'a str,
    rle_cluster_lengths: Vec<(usize, usize)>,
    len: usize,
}

// shorthand for char string for in crate usage
pub(crate) type CS<'a> = CharString<'a>;

// Rust strings vs. Python
// -----------------------
// Example string "नमस्ते" from the Rust documentation
// "नमस्ते".len() -> 18; num bytes (equal to len("नमस्ते".encode()) in Python)
// "नमस्ते".chars().count() -> 6; num unicode code points (equal to len("नमस्ते") in Python, and
// CharString::new("नमस्ते", false).len())
// CharString::new("नमस्ते", true).len() -> 4; num grapheme clusters, closest to what
// humans consider to be characters (in Python available via third party libraries)

impl<'s> CharString<'s> {
    pub fn new(str: &'s str, use_graphemes: bool) -> CharString<'s> {
        let cluster_lengths: Vec<usize> = if use_graphemes {
            str.graphemes(true).map(str::len).collect()
        } else {
            str.chars().map(char::len_utf8).collect()
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

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn get_char_byte_lengths(&self) -> Vec<usize> {
        run_length_decode(&self.rle_cluster_lengths)
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

    pub fn get(&self, n: usize) -> Option<&'s str> {
        if n >= self.len() {
            return None;
        }
        let (start, end) = self.byte_start_end(n);
        Some(&self.str[start..end])
    }

    pub fn get_char(&self, n: usize) -> Option<Character<'s>> {
        self.get(n).map(|str| Character { str })
    }

    pub fn sub(&self, start: usize, end: usize) -> &'s str {
        assert!(start <= end, "start cannot be larger than end");
        let start = start.min(self.len());
        let end = end.min(self.len());
        if self.is_empty() || start == end {
            return "";
        }
        let (start, end) = self.char_range_to_byte_range(start, end);
        &self.str[start..end]
    }

    pub fn chars(&self) -> impl Iterator<Item = Character<'_>> {
        (0..self.len()).map(|i| self.get_char(i).unwrap())
    }

    pub fn split(s: &str, use_graphemes: bool) -> impl Iterator<Item = &str> {
        let cluster_lengths: Vec<_> = if use_graphemes {
            s.graphemes(true).map(str::len).collect()
        } else {
            s.chars().map(char::len_utf8).collect()
        };
        cluster_lengths.into_iter().scan(0usize, |acc, l| {
            let s = &s[*acc..*acc + l];
            *acc += l;
            Some(s)
        })
    }
}

impl Display for CharString<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.str,)
    }
}

#[derive(Debug)]
pub struct Character<'s> {
    pub str: &'s str,
}

#[inline]
pub(crate) fn is_whitespace(s: &str) -> bool {
    s.chars().all(char::is_whitespace)
}

#[inline]
pub(crate) fn is_alphabetic(s: &str) -> bool {
    s.chars().all(char::is_alphabetic)
}

#[inline]
pub(crate) fn is_number(s: &str) -> bool {
    s.chars().all(char::is_numeric)
}

#[inline]
pub(crate) fn is_punctuation(s: &str) -> bool {
    Regex::new(r"^\p{P}+$").unwrap().is_match(s)
}

#[inline]
pub(crate) fn is_dash_punctuation(s: &str) -> bool {
    Regex::new(r"^[\p{Pd}{Pc}]+$").unwrap().is_match(s)
}

#[inline]
pub(crate) fn is_left_punctuation(s: &str) -> bool {
    Regex::new(r"^[\p{Pi}\p{Ps}]+$").unwrap().is_match(s)
}

#[inline]
pub(crate) fn is_other_punctuation(s: &str) -> bool {
    Regex::new(r"^[\p{Po}]+$").unwrap().is_match(s)
}

#[inline]
pub(crate) fn is_right_punctuation(s: &str) -> bool {
    Regex::new(r"^[\p{Pf}\p{Pe}]+$").unwrap().is_match(s)
}

impl Character<'_> {
    pub fn byte_len(&self) -> usize {
        self.str.len()
    }

    pub fn is_ascii(&self) -> bool {
        self.byte_len() == 1
    }

    pub fn is_whitespace(&self) -> bool {
        is_whitespace(self.str)
    }

    pub fn is_alphabetic(&self) -> bool {
        is_alphabetic(self.str)
    }

    pub fn is_number(&self) -> bool {
        is_number(self.str)
    }

    pub fn is_punctuation(&self) -> bool {
        is_punctuation(self.str)
    }

    pub fn is_dash_punctuation(&self) -> bool {
        is_dash_punctuation(self.str)
    }

    pub fn is_left_punctuation(&self) -> bool {
        is_left_punctuation(self.str)
    }

    pub fn is_right_punctuation(&self) -> bool {
        is_right_punctuation(self.str)
    }

    pub fn is_other_punctuation(&self) -> bool {
        is_other_punctuation(self.str)
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
    fn extract_bound(ob: &Bound<'a, PyAny>) -> PyResult<Self> {
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

impl<'py> IntoPyObject<'py> for Normalization {
    type Target = PyString;
    type Output = Bound<'py, Self::Target>;
    type Error = std::convert::Infallible;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            Normalization::NFC => "nfc",
            Normalization::NFD => "nfd",
            Normalization::NFKC => "nfkc",
            Normalization::NFKD => "nfkd",
        }
        .into_pyobject(py)
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
pub(super) fn add_submodule(py: Python<'_>, parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "unicode")?;
    m.add_function(wrap_pyfunction!(normalize, m.clone())?)?;
    parent_module.add_submodule(&m)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::unicode::{
        is_alphabetic, is_dash_punctuation, is_left_punctuation, is_number, is_other_punctuation,
        is_punctuation, is_right_punctuation, is_whitespace, CS,
    };

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
        assert_eq!(s.get(11).unwrap(), "ä");

        // test with string that has more than one code point per character:
        // the string "नमस्ते" should have 18 utf8 bytes, 6 code points, and 4 grapheme
        // clusters (characters)

        // first test with regular char string, which should behave unexpected
        let s = CS::new("नमस्ते", false);
        assert_eq!(s.str.len(), 18);
        assert_ne!(s.len(), 4);
        assert_ne!(s.get(2).unwrap(), "स्");
        assert_ne!(s.sub(2, 4), "स्ते");

        // now test with grapheme based char string, which should behave as expected
        // let s = CS::new("नमस्ते", true);
        // assert_eq!(s.str.len(), 18);
        // assert_eq!(s.len(), 3);
        // assert_eq!(s.get(2).unwrap(), "स्");
        // assert_eq!(s.sub(2, 4), "स्ते");
    }

    #[test]
    fn test_helpers() {
        assert!(is_number("1234"));
        assert!(is_alphabetic("test"));
        assert!(!["!", "1"].into_iter().any(is_alphabetic));
        assert!(is_whitespace("\n"));
        assert!(["-", "[", ",", "}"].into_iter().all(is_punctuation));
        assert!(["-"].into_iter().all(is_dash_punctuation));
        assert!(["[", "(", "{"].into_iter().all(is_left_punctuation));
        assert!(["]", ")", "}"].into_iter().all(is_right_punctuation));
        assert!(["!", "?", ",", "."].into_iter().all(is_other_punctuation));
    }
}
