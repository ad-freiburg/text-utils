use std::fs;
use std::path::PathBuf;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::distributions::WeightedIndex;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use text_utils::edit::{distance, operations};
use text_utils::prefix::PrefixTreeSearch;
use text_utils::prefix_tree::Node;
use text_utils::prefix_vec::PrefixVec;
use text_utils::text::{clean, match_words, word_boundaries};
use text_utils::tokenization::{
    token_groups_to_sparse_coo_matrix, BPETokenizer, BPETokenizerConfig, ByteGroups, ByteTokenizer,
    ByteTokenizerConfig, CharTokenizer, CharTokenizerConfig, GroupAggregation, Grouping,
    SpecialConfig, TokenGroup, Tokenize,
};
use text_utils::utils::{
    accumulate_pub, find_subsequences_of_max_size_k, run_length_decode_pub, run_length_encode_pub,
};

const INPUT_SIZES: [usize; 4] = [16, 128, 512, 2048];

fn bench_edit_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("edit_distance");
    let mut rng = ChaCha8Rng::seed_from_u64(22);
    for (idx, from_size) in INPUT_SIZES.iter().take(INPUT_SIZES.len() - 1).enumerate() {
        for to_size in INPUT_SIZES.iter().skip(idx + 1) {
            let from_str: String = (&mut rng)
                .sample_iter::<char, _>(rand::distributions::Standard)
                .take(*from_size)
                .collect();
            let to_str: String = (&mut rng)
                .sample_iter::<char, _>(rand::distributions::Standard)
                .take(*to_size)
                .collect();
            group.bench_with_input(
                BenchmarkId::new("edit_distance", format!("{} -> {}", from_size, to_size)),
                &(from_str.as_str(), to_str.as_str()),
                |b, &(from, to)| {
                    b.iter(|| distance(from, to, true, true, false, false));
                },
            );
            group.bench_with_input(
                BenchmarkId::new("edit_operations", format!("{} -> {}", from_size, to_size)),
                &(from_str.as_str(), to_str.as_str()),
                |b, &(from, to)| {
                    b.iter(|| operations(from, to, true, true, false));
                },
            );
        }
    }
    group.finish();
}

fn bench_text(c: &mut Criterion) {
    let mut group = c.benchmark_group("text");
    let mut rng = ChaCha8Rng::seed_from_u64(22);
    for size in INPUT_SIZES.iter() {
        let mut str: String = (&mut rng)
            .sample_iter::<char, _>(rand::distributions::Standard)
            .take(*size)
            .collect();
        for _ in 0..size / 4 {
            let possible_insert_indices: Vec<usize> = (0..=str.len())
                .filter(|i| str.is_char_boundary(*i))
                .collect();
            if possible_insert_indices.is_empty() {
                continue;
            }
            let insert_idx =
                possible_insert_indices[rng.gen_range(0..possible_insert_indices.len())];
            str.insert(insert_idx, ' ');
        }
        group.bench_with_input(
            BenchmarkId::new("clean", format!("{}", size)),
            str.as_str(),
            |b, str| {
                b.iter(|| clean(str, true));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("word_boundaries", format!("{}", size)),
            str.as_str(),
            |b, str| {
                b.iter(|| word_boundaries(str, true));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("match_words", format!("{}", size)),
            str.as_str(),
            |b, str| {
                b.iter(|| match_words(str, str, false));
            },
        );
    }
}

fn bench_tokenizer(c: &mut Criterion) {
    let mut group = c.benchmark_group("tokenizer");
    let mut rng = ChaCha8Rng::seed_from_u64(22);
    let char_tok = CharTokenizer::new(
        CharTokenizerConfig {
            use_graphemes: true,
        },
        SpecialConfig::default(),
        None,
    );
    let tokenize_cfg = ByteTokenizerConfig {
        use_graphemes: true,
        groups: ByteGroups::Bytes,
        aggregation: GroupAggregation::Mean,
        pad_to_multiple_of: None,
    };
    let byte_tok_byte_groups =
        ByteTokenizer::new(tokenize_cfg.clone(), SpecialConfig::default(), None);
    let byte_tok_code_point_groups =
        ByteTokenizer::new(tokenize_cfg, SpecialConfig::default(), None);
    let dir = env!("CARGO_MANIFEST_DIR");
    let bpe_tok = BPETokenizer::new(
        BPETokenizerConfig {
            use_graphemes: true,
            merge_file: PathBuf::from(dir).join("data/multi/bpe_multi_16384_3.merges"),
            max_vocab_size: None,
        },
        SpecialConfig::default(),
        None,
    )
    .expect("failed to create bpe tokenizer");
    let multi30k = fs::read_to_string(PathBuf::from(dir).join("resources/test/multi30k.txt"))
        .expect("failed to read file")
        .replace("\n", " ");
    for size in INPUT_SIZES.iter() {
        let str = multi30k[..*size].to_string();
        group.bench_with_input(
            BenchmarkId::new("char", format!("{}", size)),
            str.as_str(),
            |b, str| {
                b.iter(|| char_tok.tokenize(str, None, None, None, true));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("byte (byte groups)", format!("{}", size)),
            str.as_str(),
            |b, str| {
                b.iter(|| byte_tok_byte_groups.tokenize(str, None, None, None, true));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("byte (code point groups)", format!("{}", size)),
            str.as_str(),
            |b, str| {
                b.iter(|| byte_tok_code_point_groups.tokenize(str, None, None, None, true));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("bpe", format!("{}", size)),
            str.as_str(),
            |b, str| {
                b.iter(|| bpe_tok.tokenize(str, None, None, None, true));
            },
        );

        let grouping: Grouping = (
            (&mut rng)
                .sample_iter::<usize, _>(rand::distributions::Uniform::from(1..5))
                .take(*size)
                .map(|l| TokenGroup::Full(l))
                .collect(),
            GroupAggregation::Mean,
        );
        let sizes = vec![grouping.0.iter().map(|g| g.len()).sum::<usize>(); 32];

        group.bench_with_input(
            BenchmarkId::new("sparse_coo_single_stage", format!("{size}")),
            &(&grouping, &sizes[..1]),
            |b, (grouping, sizes)| b.iter(|| token_groups_to_sparse_coo_matrix(&[grouping], sizes)),
        );

        let groupings = vec![&grouping; 32];
        group.bench_with_input(
            BenchmarkId::new("sparse_coo_single_stage_batched_32", format!("{size}")),
            &(&groupings, &sizes),
            |b, (groupings, sizes)| b.iter(|| token_groups_to_sparse_coo_matrix(&groupings, sizes)),
        );
        drop(groupings);
    }
}

fn bench_utils(c: &mut Criterion) {
    let mut group = c.benchmark_group("utils");
    let mut rng = ChaCha8Rng::seed_from_u64(22);
    let dist: WeightedIndex<usize> = WeightedIndex::new([16, 4, 2, 1]).unwrap();
    for size in INPUT_SIZES.iter() {
        let input: Vec<usize> = (&mut rng)
            .sample_iter(&dist)
            .take(*size)
            .map(|idx| idx + 1)
            .collect();
        group.bench_with_input(
            BenchmarkId::new("accumulate", format!("{size}")),
            &input,
            |b, input| {
                b.iter(|| accumulate_pub(input));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("run_length_encode", format!("{size}")),
            &input,
            |b, input| {
                b.iter(|| run_length_encode_pub(input));
            },
        );
        let input_encoded = run_length_encode_pub(&input);
        group.bench_with_input(
            BenchmarkId::new("run_length_decode", format!("{size}")),
            &input_encoded,
            |b, input| {
                b.iter(|| run_length_decode_pub(input));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("subsequences of max size", format!("{size}")),
            &input,
            |b, input| {
                b.iter(|| {
                    find_subsequences_of_max_size_k(input, *size, |values| values.iter().sum())
                })
            },
        );
    }
}

fn bench_prefix(c: &mut Criterion) {
    let dir = env!("CARGO_MANIFEST_DIR");
    let multi30k = fs::read_to_string(PathBuf::from(dir).join("resources/test/multi30k.txt"))
        .expect("failed to read file")
        .replace("\n", " ");
    let words: Vec<_> = multi30k
        .split_whitespace()
        .map(|s| s.as_bytes().to_vec())
        .collect();
    let mut rng = ChaCha8Rng::seed_from_u64(22);
    // sample random word from all words
    let word = words.choose(&mut rng).unwrap().as_slice();
    let mut group = c.benchmark_group("prefix");

    // benchmark prefix tree
    let mut tree: Node<_> = words.iter().cloned().zip(0..words.len()).collect();
    group.bench_with_input("tree_build", &words, |b, input| {
        b.iter(|| {
            let _tree: Node<_> = input.iter().cloned().zip(0..input.len()).collect();
        });
    });
    group.bench_with_input("tree_insert", word, |b, input| {
        b.iter(|| tree.insert(input, 1));
    });
    group.bench_with_input("tree_get", word, |b, input| {
        b.iter(|| tree.get(input));
    });

    // benchmark prefix vec
    let mut vec: PrefixVec<_> = words.iter().cloned().zip(0..words.len()).collect();
    group.bench_with_input("vec_build", &words, |b, input| {
        b.iter(|| {
            let _vec: PrefixVec<_> = input.iter().cloned().zip(0..input.len()).collect();
        });
    });
    for size in vec![128, 256, 512, 1024, 2048, 4096, 8192] {
        group.bench_with_input(format!("vec_build_{size}"), &words[..size], |b, input| {
            b.iter(|| {
                let _vec: PrefixVec<_> = input.iter().cloned().zip(0..input.len()).collect();
            });
        });
    }
    group.bench_with_input("vec_insert", word, |b, input| {
        b.iter(|| vec.insert(input, 1));
    });
    group.bench_with_input("vec_get", word, |b, input| {
        b.iter(|| vec.get(input));
    });
    vec.compute_memo(2);
    group.bench_with_input("vec_get_memo_2", word, |b, input| {
        b.iter(|| vec.get(input));
    });
}

criterion_group!(
    benches,
    bench_edit_distance,
    bench_text,
    bench_tokenizer,
    bench_utils,
    bench_prefix
);
criterion_main!(benches);
