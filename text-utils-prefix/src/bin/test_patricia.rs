use std::{fs, path::PathBuf};

use text_utils_prefix::PatriciaTrie;

fn main() {
    let dir = env!("CARGO_MANIFEST_DIR");
    let index = fs::read_to_string(PathBuf::from(dir).join("resources/test/index.txt"))
        .expect("failed to read file");
    let n = 10_000_000;

    let trie: PatriciaTrie<_> = index
        .lines()
        .enumerate()
        .map(|(i, s)| (s.as_bytes().to_vec(), i))
        .take(n)
        .collect();
    let stats = trie.stats();
    println!("{stats:#?}");
}
