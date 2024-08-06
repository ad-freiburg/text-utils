use std::{fs, path::PathBuf};

use text_utils_prefix::{AdaptiveRadixTrie, ArtContinuationTrie};

fn main() {
    let dir = env!("CARGO_MANIFEST_DIR");
    let index = fs::read_to_string(PathBuf::from(dir).join("resources/test/index.txt"))
        .expect("failed to read file");
    let n = 100_000_000;

    let art: AdaptiveRadixTrie<_> = index
        .lines()
        .enumerate()
        .map(|(i, s)| (s.as_bytes().to_vec(), i))
        .take(n)
        .collect();
    let stats = art.stats();
    println!("{stats:#?}");

    let cont: ArtContinuationTrie<_> = index
        .lines()
        .enumerate()
        .map(|(i, s)| (s.as_bytes().to_vec(), i))
        .take(n)
        .collect();
    let start = std::time::Instant::now();
    let _ = cont.sub_index_by_values(0..10000usize);
    println!(
        "sub_index_by_values took: {:?}ms",
        start.elapsed().as_micros() / 1000
    );
}
