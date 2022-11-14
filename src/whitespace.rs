use itertools::Itertools;

pub fn remove(s: &str) -> String {
    s.split_whitespace().join("")
}

pub fn full(s: &str) -> String {
    s.split_whitespace().map(|w| w.chars().join(" ")).join(" ")
}

pub fn operations(from: &str, to: &str) -> Vec<usize> {
    assert_eq!(remove(from), remove(to), "from and to should only differ in whitespaces");
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
            from_ptr += 1;
            to_ptr += 1;
        } else if to_char.is_some() && to_char.unwrap().is_whitespace() {
            operations.push(1);
            from_ptr += 1;
            to_ptr += 1;
        } else if from_char.is_whitespace() {
            operations.push(2);
            from_ptr += 1;
        } else {
            panic!("should not happen: {}, {:?}", from_char, to_char);
        }
    }
    operations
}

#[cfg(test)]
mod tests {
    use crate::whitespace::{full, operations, remove};

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
        let from = " t h isis a test  ";
        let to = "this is a test";
        assert_eq!(
            operations(from, from),
            vec![0; from.chars().count()]
        );
        assert_eq!(
            operations(to, to),
            vec![0; to.chars().count()]
        );
        assert_eq!(
            operations(from, to),
            vec![2, 0, 2, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2]
        );
    }
}
