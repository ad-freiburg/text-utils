use itertools::Itertools;
use regex::{escape, Regex};
use crate::text::clean;

pub fn remove(s: &str) -> String {
    s.split_whitespace().join("")
}

pub fn full(s: &str) -> String {
    s.split_whitespace().map(|w| w.chars().join(" ")).join(" ")
}

pub fn operations(from: &str, to: &str) -> Vec<usize> {
    assert_eq!(
        remove(from),
        remove(to),
        "from and to should only differ in whitespaces"
    );
    let from_chars: Vec<char> = from.chars().collect();
    let to_chars: Vec<char> = to.chars().collect();
    let mut operations = vec![];
    operations.reserve(from_chars.len().max(to_chars.len()));
    let mut from_ptr = 0;
    let mut to_ptr = 0;
    while from_ptr < from_chars.len() {
        let from_char = from_chars[from_ptr];
        let to_char = if to_ptr < to_chars.len() { Some(to_chars[to_ptr]) } else { None };
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

pub fn repair(s: &str, operations: &[usize]) -> String {
    let chars: Vec<char> = s.chars().collect();
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
        assert!(
            *op == 0 || *op == 1 || *op == 2,
            "operation should be either 0, 1, or 2, but got {}",
            op
        );
        let prev_char = if idx > 0 {
            chars[idx - 1]
        } else {
            '#'
        };
        if *op == 1 && !prev_char.is_whitespace() && !char.is_whitespace() {
            new_chars.push(' ');
            new_chars.push(*char);
        } else if *op == 2 && char.is_whitespace() {
            continue;
        } else {
            new_chars.push(*char);
        }
    }
    clean(new_chars.iter().join("").as_str())
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
        assert_eq!(full(" t   h is is \n\t a tes    t "), "t h i s i s a t e s t");
        assert_eq!(full(""), "");
    }

    #[test]
    fn test_operations() {
        let from = " t  h isis a test  ";
        let to = "this is a test";
        assert_eq!(operations(from, from), vec![0; from.chars().count()]);
        assert_eq!(operations(to, to), vec![0; to.chars().count()]);
        assert_eq!(
            operations(from, to),
            vec![2, 0, 2, 2, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2]
        );
    }

    #[test]
    fn test_repair() {
        let from = " t h isis a test  ";
        let to = "this is a test";
        assert_eq!(repair(from, &operations(from, to)), to);
        assert_eq!(repair(to, &operations(to, from)), "t h isis a test");
        assert_eq!(repair("    ", &vec![2, 2, 2, 2]), "");
        assert_eq!(repair("  t  ", &vec![0, 2, 0, 0, 1]), "t");
        assert_eq!(repair("", &vec![]), "");
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
