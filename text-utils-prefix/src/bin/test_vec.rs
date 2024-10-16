use std::{fs, path::PathBuf};

use text_utils_prefix::PrefixVec;

fn main() {
    let dir = env!("CARGO_MANIFEST_DIR");
    let index = fs::read_to_string(PathBuf::from(dir).join("resources/test/index.txt"))
        .expect("failed to read file");
    let n = 10_000_000;
    let words: Vec<_> = index.lines().map(|s| s.as_bytes()).take(n).collect();

    let _trie: PrefixVec<_> = words.iter().enumerate().map(|(i, w)| (w, i)).collect();
    // let stats = trie.stats();
    // println!("{stats:#?}");
}
