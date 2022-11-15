use std::fmt::{Display, Formatter};
use unicode_segmentation::{UnicodeSegmentation};

#[derive(Debug, Clone)]
pub struct GraphemeString {
    s: String,
    cum_cluster_lengths: Vec<usize>,
}

// Rust strings vs. Python
// -----------------------
// Example string "नमस्ते" from the Rust documentation
// "नमस्ते".len() -> 18; num bytes (equal to len("नमस्ते".encode()) in Python)
// "नमस्ते".chars().count() -> 6; num unicode code units (equal to len("नमस्ते") in Python)
// GraphemeString::from("नमस्ते").len() -> 4; num grapheme clusters, closest to what
// humans consider to be characters (in Python available via third party libraries)

impl GraphemeString {
    pub fn as_str(&self) -> &str {
        self.s.as_str()
    }

    pub fn byte_len(&self) -> usize {
        self.s.len()
    }

    pub fn byte_lengths(&self) -> &Vec<usize> {
        &self.cum_cluster_lengths
    }

    pub fn len(&self) -> usize {
        self.cum_cluster_lengths.len()
    }

    pub fn get(&self, n: usize) -> &str {
        assert!(n < self.len());
        let start = if n == 0 { 0 } else { self.cum_cluster_lengths[n - 1] };
        &self.s[start..self.cum_cluster_lengths[n]]
    }

    pub fn sub(&self, start: usize, end: usize) -> &str {
        assert!(start <= end && end <= self.len());
        if self.len() == 0 || start == end {
            return "";
        }
        let start = if start == 0 { 0 } else { self.cum_cluster_lengths[start - 1] };
        let end = self.cum_cluster_lengths[end - 1];
        &self.s[start..end]
    }
}

impl Display for GraphemeString {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GraphemeString(\"{}\", bytes={}, clusters={})",
            self.s,
            self.s.len(),
            self.len()
        )
    }
}

impl From<String> for GraphemeString {
    fn from(s: String) -> Self {
        let mut cum_cluster_lengths: Vec<usize> = s
            .grapheme_indices(true)
            .map(|(len, _)| len)
            .collect();
        if !s.is_empty() {
            cum_cluster_lengths.push(s.len());
            cum_cluster_lengths.remove(0);
        }
        GraphemeString {
            s,
            cum_cluster_lengths
        }
    }
}

impl From<&str> for GraphemeString {
    fn from(s: &str) -> Self {
        s.to_string().into()
    }
}

#[cfg(test)]
mod tests {
    use crate::unicode::GraphemeString;

    #[test]
    fn test_grapheme_string() {
        // test with empty string
        let s: GraphemeString = "".into();
        assert_eq!(s.byte_len(), 0);
        assert_eq!(s.len(), 0);
        assert_eq!(s.sub(0, s.len()), "");
        assert_eq!(s.cum_cluster_lengths, vec![]);
        // test with ascii string
        let s: GraphemeString = "this is a test".into();
        assert_eq!(s.byte_len(), 14);
        assert_eq!(s.len(), 14);
        assert_eq!(s.sub(10, 14), "test");
        assert_eq!(s.sub(10, 10), "");
        let mut cum_cluster_lengths = (1..=s.len()).collect::<Vec<usize>>();
        assert_eq!(s.cum_cluster_lengths, cum_cluster_lengths);
        // test with non ascii string
        let s: GraphemeString = "this is a täst".into();
        assert_eq!(s.byte_len(), 15);
        assert_eq!(s.len(), 14);
        assert_eq!(s.sub(10, 14), "täst");
        cum_cluster_lengths[11] = 13;
        cum_cluster_lengths[12] = 14;
        cum_cluster_lengths[13] = 15;
        assert_eq!(s.cum_cluster_lengths, cum_cluster_lengths);
        assert_eq!(s.get(11), "ä");
        // test with string that has more than one code point per character:
        // the string below should have 18 utf8 bytes, 6 code points, and 4 grapheme
        // clusters (characters)
        let s: GraphemeString = "नमस्ते".into();
        assert_eq!(s.byte_len(), 18);
        assert_eq!(s.len(), 4);
        assert_eq!(s.get(2), "स्");
        assert_eq!(s.sub(2, s.len()), "स्ते");
    }
}
