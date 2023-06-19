use std::{
    path::PathBuf,
    thread::sleep,
    time::{Duration, Instant},
};

use text_correction_utils::{
    data::{
        labeling::LabelingConfig,
        loading::{
            text_data_generator_from_files, BatchLimitType, BatchedIterator, BufferedIterator,
            ItemSize, PipelineIterator, TensorizedIterator, TextIterationStrategy, TextIterator,
        },
        postprocessing::PostprocessingFnConfig,
        preprocessing::{PreprocessingFnConfig, SpellingCorruptionMode},
        text_data_pipeline_with_tokenizer, PostprocessingConfig, PreprocessingConfig, TextDataInfo,
        TextDataPipelineConfig,
    },
    text::file_size,
    tokenization::{
        ByteGroups, ByteTokenizerConfig, GroupAggregation, SpecialConfig, TokenizeConfig,
        TokenizerConfig, BOS, EOS, PAD,
    },
    unicode::Normalization,
    utils::progress_bar,
};

use clap::Parser;

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long, num_args(1..))]
    files: Vec<String>,

    #[arg(short, long, default_value_t = 4)]
    num_threads: u8,

    #[arg(short, long, default_value_t = 65536)]
    batch_limit: usize,

    #[arg(short, long, default_value_t = 10000)]
    limit: usize,

    #[arg(short, long, default_value_t = 32)]
    buffer_size: usize,

    #[arg(short, long, default_value_t = 32)]
    prefetch_factor: usize,

    #[arg(short, long, default_value_t = 10)]
    delay_ms: u64,

    #[arg(short, long)]
    char_file: Option<PathBuf>,

    #[arg(short, long, default_value_t = 512)]
    max_length: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    assert!(
        !args.files.is_empty(),
        "expected at least one file for preprocessing"
    );

    let tokenizer_cfg = TokenizerConfig {
        tokenize: TokenizeConfig::Byte(ByteTokenizerConfig {
            groups: ByteGroups::CodePoints,
            aggregation: GroupAggregation::Mean,
            use_graphemes: true,
            pad_to_multiple_of: Some(32),
        }),
        special: SpecialConfig {
            pad: PAD.into(),
            prefix: vec![BOS.into()],
            suffix: vec![EOS.into()],
            tokens: vec![BOS.into(), EOS.into(), PAD.into()],
        },
        language: None,
    };
    let pipeline_cfg = TextDataPipelineConfig {
        preprocessing: PreprocessingConfig::Single(PreprocessingFnConfig::Chain(vec![
            PreprocessingFnConfig::Clean(true),
            PreprocessingFnConfig::Normalize(Normalization::NFKC, true),
            PreprocessingFnConfig::ByteSubstring(512, true),
            PreprocessingFnConfig::SpellingCorruption(
                0.2,
                true,
                SpellingCorruptionMode::Artificial(0.25, 2.0, args.char_file),
            ),
            PreprocessingFnConfig::WhitespaceCorruption(0.05, 0.2, true),
        ])),
        labeling: LabelingConfig::SequenceGeneration(tokenizer_cfg.clone()),
        postprocessing: PostprocessingConfig::Single(PostprocessingFnConfig::None),
    };
    let (pipeline, _max_length) =
        text_data_pipeline_with_tokenizer(pipeline_cfg, tokenizer_cfg.clone(), args.max_length)?;

    let seed = 22;
    let num_lines: usize = args
        .files
        .iter()
        .filter_map(|f| file_size(f).ok())
        .map(|(lines, _)| lines)
        .sum();
    let generators = args
        .files
        .iter()
        .map(|f| text_data_generator_from_files(f, None, None))
        .collect::<anyhow::Result<_>>()?;
    let text_iter = TextIterator::new(generators, TextIterationStrategy::Weighted, Some(seed))?;

    let mut iter = text_iter
        .enumerate()
        .filter_map(move |(item_idx, (d, file_idx))| {
            if let Ok(d) = d {
                Some((
                    d,
                    TextDataInfo {
                        file_idx,
                        seed: seed + item_idx as u64,
                        ..Default::default()
                    },
                ))
            } else {
                None
            }
        })
        .take(args.limit)
        .pipe(pipeline, args.num_threads)
        .filter_map(|i| i.ok())
        .batched(
            true,
            true,
            args.prefetch_factor,
            args.batch_limit,
            BatchLimitType::PaddedItemSize,
            Some(seed),
        )
        .tensorized(tokenizer_cfg)?
        .buffered(args.buffer_size);

    let start = Instant::now();
    let mut load_times = vec![];
    let mut sizes = vec![];
    let mut ratios = vec![];
    let p_bar = progress_bar("preprocessing", num_lines.min(args.limit) as u64, false);
    loop {
        let load_batch = Instant::now();
        if let Some(item) = iter.next() {
            load_times.push(load_batch.elapsed().as_secs_f64());
            sizes.push(item.0.len());
            let lengths: Vec<_> = item.0.iter().map(|item| item.size()).collect();
            ratios.push(
                lengths.iter().max().copied().unwrap_or(0) as f64
                    / lengths.into_iter().min().unwrap_or(1) as f64,
            );
            p_bar.inc(item.0.len() as u64);
        } else {
            break;
        };
        // simulates model running
        sleep(Duration::from_millis(args.delay_ms));
    }
    p_bar.finish_and_clear();
    println!(
        "average batch load time: {:.2}ms",
        1000.0 * load_times.iter().sum::<f64>() / load_times.len() as f64
    );
    println!(
        "average batch size: {:.2}",
        sizes.iter().sum::<usize>() as f64 / sizes.len() as f64
    );
    println!(
        "average max/min ratio: {:.2}",
        ratios.iter().sum::<f64>() / sizes.len() as f64
    );
    println!("preprocessing took {:.2}s", start.elapsed().as_secs_f64());

    Ok(())
}
