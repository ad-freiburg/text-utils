use std::fmt::{Display, Formatter};
use std::str::Chars;
use unicode_segmentation::{UnicodeSegmentation};
use crate::utils::accumulate;

#[derive(Debug, Clone)]
pub struct CharString<'a> {
    pub str: &'a str,
    pub(crate) cluster_lengths: Vec<usize>,
    pub(crate) cum_cluster_lengths: Vec<usize>,
}

// shorthand for grapheme string for in crate usage
pub(crate) type CS<'a> = CharString<'a>;

// Rust strings vs. Python
// -----------------------
// Example string "नमस्ते" from the Rust documentation
// "नमस्ते".len() -> 18; num bytes (equal to len("नमस्ते".encode()) in Python)
// "नमस्ते".chars().count() -> 6; num unicode code units (equal to len("नमस्ते") in Python)
// GraphemeString::from("नमस्ते").len() -> 4; num grapheme clusters, closest to what
// humans consider to be characters (in Python available via third party libraries)

impl CharString<'_> {
    pub fn new(str: &str, use_graphemes: bool) -> CharString {
        let cluster_lengths: Vec<usize>;
        if use_graphemes {
            cluster_lengths = str
                .graphemes(true)
                .map(|s| s.len())
                .collect();
        } else {
            cluster_lengths = str
                .chars()
                .map(|c| c.len_utf8())
                .collect();
        }
        let cum_cluster_lengths = accumulate(&cluster_lengths);
        CharString {
            str,
            cluster_lengths,
            cum_cluster_lengths,
        }
    }

    pub fn byte_len(&self) -> usize {
        self.str.len()
    }

    pub fn len(&self) -> usize {
        self.cum_cluster_lengths.len()
    }

    pub fn get(&self, n: usize) -> &str {
        assert!(n < self.len());
        let start = if n == 0 { 0 } else { self.cum_cluster_lengths[n - 1] };
        &self.str[start..self.cum_cluster_lengths[n]]
    }

    pub fn sub(&self, start: usize, end: usize) -> &str {
        assert!(start <= end && end <= self.len());
        if self.len() == 0 || start == end {
            return "";
        }
        let start = self.cum_cluster_lengths[start] - self.cluster_lengths[start];
        let end = self.cum_cluster_lengths[end - 1];
        &self.str[start..end]
    }

    pub fn chars(&self) -> Characters {
        Characters {
            str: self.str,
            cum_cluster_lengths: &self.cum_cluster_lengths,
            idx: 0,
        }
    }

    pub fn char_byte_lengths(&self) -> &Vec<usize> {
        &self.cluster_lengths
    }

    pub fn cum_char_byte_lengths(&self) -> &Vec<usize> {
        &self.cum_cluster_lengths
    }
}

pub struct Characters<'a> {
    str: &'a str,
    cum_cluster_lengths: &'a Vec<usize>,
    idx: usize,
}

impl<'a> Iterator for Characters<'a> {
    type Item = Character<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.cum_cluster_lengths.len() {
            return None;
        }
        let start = if self.idx == 0 { 0 } else { self.cum_cluster_lengths[self.idx - 1] };
        let char = Character {
            str: &self.str[start..self.cum_cluster_lengths[self.idx]]
        };
        self.idx += 1;
        Some(char)
    }
}

pub struct Character<'a> {
    pub str: &'a str,
}

impl Character<'_> {
    pub fn byte_len(&self) -> usize {
        self.str.len()
    }

    pub fn is_ascii(&self) -> bool {
        self.byte_len() == 1
    }

    pub fn is_whitespace(&self) -> bool {
        self.str.chars().all(|c| c.is_whitespace())
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

impl Display for CharString<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GraphemeString(\"{}\", bytes={}, clusters={})",
            self.str,
            self.str.len(),
            self.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::unicode::CS;

    #[test]
    fn test_grapheme_string() {
        // test with empty string
        let s = CS::new("", false);
        assert_eq!(s.str.len(), 0);
        assert_eq!(s.len(), 0);
        assert_eq!(s.sub(0, s.len()), "");
        let empty: Vec<usize> = Vec::new();
        assert_eq!(s.cum_cluster_lengths, empty);
        // test with ascii string
        let s = CS::new("this is a test", false);
        assert_eq!(s.str.len(), 14);
        assert_eq!(s.len(), 14);
        assert_eq!(s.sub(10, 14), "test");
        assert_eq!(s.sub(10, 10), "");
        let mut cum_cluster_lengths = (1..=s.len()).collect::<Vec<usize>>();
        assert_eq!(s.cum_cluster_lengths, cum_cluster_lengths);
        // test with non ascii string
        let s = CS::new("this is a täst", false);
        assert_eq!(s.str.len(), 15);
        assert_eq!(s.len(), 14);
        assert_eq!(s.sub(10, 14), "täst");
        cum_cluster_lengths[11] = 13;
        cum_cluster_lengths[12] = 14;
        cum_cluster_lengths[13] = 15;
        assert_eq!(s.cum_cluster_lengths, cum_cluster_lengths);
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
