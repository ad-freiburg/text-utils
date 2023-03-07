use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::distributions::WeightedIndex;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use text_correction_utils::edit::{distance, operations};
use text_correction_utils::text::{clean, match_words, word_boundaries};
use text_correction_utils::tokenization::{
    token_groups_to_sparse_coo_matrix, ByteGroups, ByteTokenizer, ByteTokenizerConfig,
    CharTokenizer, CharTokenizerConfig, GroupAggregation, Grouping, SpecialConfig, Tokenize,
};
use text_correction_utils::utils::{
    accumulate_pub, find_subsequences_of_max_size_k, run_length_decode_pub, run_length_encode_pub,
};

const INPUT_SIZES: [usize; 3] = [16, 128, 512];

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
    };
    let byte_tok_byte_groups =
        ByteTokenizer::new(tokenize_cfg.clone(), SpecialConfig::default(), None);
    let byte_tok_code_point_groups =
        ByteTokenizer::new(tokenize_cfg, SpecialConfig::default(), None);
    for size in INPUT_SIZES.iter() {
        let str: String = (&mut rng)
            .sample_iter::<char, _>(rand::distributions::Standard)
            .take(*size)
            .collect();
        group.bench_with_input(
            BenchmarkId::new("char", format!("{}", size)),
            str.as_str(),
            |b, str| {
                b.iter(|| char_tok.tokenize(str, None));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("byte (byte groups)", format!("{}", size)),
            str.as_str(),
            |b, str| {
                b.iter(|| byte_tok_byte_groups.tokenize(str, None));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("byte (code point groups)", format!("{}", size)),
            str.as_str(),
            |b, str| {
                b.iter(|| byte_tok_code_point_groups.tokenize(str, None));
            },
        );

        let mut grouping: Grouping = (
            vec![(&mut rng)
                .sample_iter::<usize, _>(rand::distributions::Uniform::from(1..5))
                .take(*size)
                .collect()],
            GroupAggregation::Mean,
        );
        let mut sizes = vec![grouping.0[0].iter().sum::<usize>()];

        group.bench_with_input(
            BenchmarkId::new("sparse_coo_single_stage", format!("{size}")),
            &(&grouping, &sizes),
            |b, (grouping, sizes)| b.iter(|| token_groups_to_sparse_coo_matrix(&[grouping], sizes)),
        );

        let second_stage = vec![4; *size / 4];
        grouping.0.push(second_stage);

        group.bench_with_input(
            BenchmarkId::new("sparse_coo_two_stage", format!("{size}")),
            &(&grouping, &sizes),
            |b, (grouping, sizes)| b.iter(|| token_groups_to_sparse_coo_matrix(&[grouping], sizes)),
        );

        let third_stage = vec![2; *size / 8];
        grouping.0.push(third_stage);

        group.bench_with_input(
            BenchmarkId::new("sparse_coo_three_stage", format!("{size}")),
            &(&grouping, &sizes),
            |b, (grouping, sizes)| b.iter(|| token_groups_to_sparse_coo_matrix(&[grouping], sizes)),
        );

        // test batched implementation with batch size 32
        let groupings: Vec<&Grouping> = (0..32).map(|_| &grouping).collect();
        sizes.append(&mut vec![sizes[0]; 31]);

        group.bench_with_input(
            BenchmarkId::new("sparse_coo_three_stage_batched_32", format!("{size}")),
            &(&groupings, &sizes),
            |b, (groupings, sizes)| b.iter(|| token_groups_to_sparse_coo_matrix(&groupings, sizes)),
        );
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

criterion_group!(
    benches,
    bench_edit_distance,
    bench_text,
    bench_tokenizer,
    bench_utils
);
criterion_main!(benches);
