use std::fs;
use std::path::PathBuf;

use criterion::{criterion_group, criterion_main, Criterion};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use text_utils_prefix::{optimized_continuations, ContinuationSearch, PrefixSearch};
use text_utils_prefix::{patricia_trie::PatriciaTrie, trie::Trie};

use art_tree::{Art, ByteString};
use patricia_tree::PatriciaMap;

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

    let continuations_json = fs::read(PathBuf::from(dir).join("resources/test/continuations.json"))
        .expect("failed to read file");
    // use serde to deserialize continuations array from json
    let continuations: Vec<String> = serde_json::from_slice(&continuations_json).unwrap();
    let (permutation, skips) = optimized_continuations(&continuations);

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
    group.bench_with_input(
        "patricia_trie_continuations",
        &("Albert", &continuations),
        |b, input| {
            let (word, continuations) = input;
            b.iter(|| trie.contains_continuations(&word, &continuations));
        },
    );
    group.bench_with_input(
        "patricia_trie_continuations_optimized",
        &("Albert", &continuations),
        |b, input| {
            let (word, continuations) = input;
            b.iter(|| {
                trie.contains_continuations_optimized(&word, &continuations, &permutation, &skips)
            });
        },
    );
    group.bench_with_input(
        "patricia_trie_continuations_batch",
        &(["Albert"; 64], &continuations),
        |b, input| {
            let (words, continuations) = input;
            b.iter(|| trie.batch_contains_continuations(words, &continuations));
        },
    );
    group.bench_with_input(
        "patricia_trie_continuations_batch_optimized",
        &(["Albert"; 64], &continuations),
        |b, input| {
            let (words, continuations) = input;
            b.iter(|| {
                trie.batch_contains_continuations_optimized(
                    words,
                    &continuations,
                    &permutation,
                    &skips,
                )
            });
        },
    );
    let inputs: [_; 64] = std::array::from_fn(|_| b"Albert".to_vec());
    let continuations: Vec<_> = continuations
        .iter()
        .map(|s| s.as_bytes().to_vec())
        .collect();
    group.bench_with_input(
        "patricia_trie_continuations_batch_optimized_parallel",
        &(inputs, continuations),
        |b, input| {
            let (words, continuations) = input;
            b.iter(|| {
                trie.batch_contains_continuations_optimized_parallel(
                    words,
                    continuations,
                    &permutation,
                    &skips,
                )
            });
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
