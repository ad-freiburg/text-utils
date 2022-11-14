use itertools::Itertools;

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

pub fn repair(s: &str, operations: &Vec<usize>) -> String {
    let chars: Vec<char> = s.chars().collect();
    let num_chars = chars.len();

    let mut new_chars = vec![];
    new_chars.reserve(num_chars);
    for (idx, (char, op)) in chars
        .iter()
        .zip(operations.iter())
        .enumerate() {
        assert!(
            *op == 0 || *op == 1 || *op == 2,
            "operation should be either 0, 1, or 2, but got {}",
            op
        );
        let prev_char = if idx == 0 { '#' } else { chars[idx - 1] };
        if *op == 1 && !prev_char.is_whitespace() && !char.is_whitespace() {
            new_chars.push(' ');
            new_chars.push(*char);
        } else if *op == 2 && char.is_whitespace() {
            continue;
        } else {
            new_chars.push(*char);
        }
    }
    new_chars.iter().join("")
}

#[cfg(test)]
mod tests {
    use crate::whitespace::{full, operations, remove, repair};

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
        assert_eq!(repair(to, &operations(to, from)), from);
        assert_eq!(repair("", &vec![1, 1, 1, 1]), "    ");
        assert_eq!(repair("    ", &vec![2, 2, 2, 2]), "");
        assert_eq!(repair("", &vec![]), "");
    }
}
