use std::fs;
use std::path::PathBuf;

use criterion::{criterion_group, criterion_main, Criterion};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use text_utils_prefix::{patricia_trie::PatriciaTrie, trie::Trie};
use text_utils_prefix::{ContinuationSearch, PrefixSearch};

use art_tree::{Art, ByteString};
use patricia_tree::PatriciaMap;

const ASCII_LETTERS: &[u8; 52] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

fn bench_prefix(c: &mut Criterion) {
    let dir = env!("CARGO_MANIFEST_DIR");
    let index = fs::read_to_string(PathBuf::from(dir).join("resources/test/index.txt"))
        .expect("failed to read file");
    let words: Vec<_> = index.lines().map(|s| s.as_bytes()).take(100_000).collect();
    let mut rng = ChaCha8Rng::seed_from_u64(22);
    // sample random word from all words
    let word = *words.choose(&mut rng).unwrap();
    println!("choose word {}", String::from_utf8_lossy(word));
    let mut group = c.benchmark_group("prefix_search");
    let continuations: Vec<_> = ASCII_LETTERS.iter().map(|&c| [c]).collect();

    // benchmark art-tree
    let mut trie: Art<_, _> = Art::new();
    for (i, word) in words.iter().enumerate() {
        trie.insert(ByteString::new(word), i);
    }
    group.bench_with_input("art_tree_insert", word, |b, input| {
        b.iter(|| trie.insert(ByteString::new(input), 1));
    });
    group.bench_with_input("art_tree_get", word, |b, input| {
        b.iter(|| trie.get(&ByteString::new(input)));
    });

    // benchmark patricia_tree
    let mut trie: PatriciaMap<_> = PatriciaMap::new();
    for (i, word) in words.iter().enumerate() {
        trie.insert(word, i);
    }
    group.bench_with_input("patricia_tree_insert", word, |b, input| {
        b.iter(|| trie.insert(input, 1));
    });
    group.bench_with_input("patricia_tree_get", word, |b, input| {
        b.iter(|| trie.get(input));
    });

    // benchmark prefix tries
    let mut trie: Trie<_> = words.iter().zip(0..words.len()).collect();
    group.bench_with_input("trie_insert", word, |b, input| {
        b.iter(|| trie.insert(input, 1));
    });
    group.bench_with_input("trie_get", word, |b, input| {
        b.iter(|| trie.get(input));
    });
    group.bench_with_input("trie_contains", word, |b, input| {
        b.iter(|| trie.contains_prefix(&input[..input.len().saturating_sub(3)]));
    });

    // benchmark patricia trie
    let mut trie: PatriciaTrie<_> = words.iter().zip(0..words.len()).collect();
    group.bench_with_input("patricia_trie_insert", word, |b, input| {
        b.iter(|| trie.insert(input, 1));
    });
    group.bench_with_input("patricia_trie_get", word, |b, input| {
        b.iter(|| trie.get(input));
    });
    group.bench_with_input("patricia_trie_contains", word, |b, input| {
        b.iter(|| trie.contains_prefix(&input[..input.len().saturating_sub(3)]));
    });
    let conts = trie.contains_continuations("Albert", &continuations);
    assert_eq!(
        conts.iter().map(|&b| if b { 1 } else { 0 }).sum::<usize>(),
        4
    );
    group.bench_with_input(
        "patricia_trie_continuations",
        &("Albert", &continuations),
        |b, input| {
            let (word, continuations) = input;
            b.iter(|| trie.contains_continuations(&word, &continuations));
        },
    );
    group.bench_with_input(
        "patricia_trie_batch_continuations",
        &(["Albert"; 64], &continuations),
        |b, input| {
            let (words, continuations) = input;
            b.iter(|| trie.batch_contains_continuations(words, &continuations));
        },
    );

    // benchmark build, load, and save
    drop(group);
    let mut group = c.benchmark_group("prefix_io");
    let n = 10_000;

    // let trie: RadixTrie<_> = words.iter().zip(0..words.len()).take(n).collect();
    // let path = PathBuf::from(dir).join("resources/test/byte_trie.bin");
    // group.bench_with_input("byte_trie_build", &words, |b, input| {
    //     b.iter(|| {
    //         input
    //             .iter()
    //             .zip(0..input.len())
    //             .take(n)
    //             .collect::<RadixTrie<_>>()
    //     });
    // });
}

criterion_group!(benches, bench_prefix);
criterion_main!(benches);
