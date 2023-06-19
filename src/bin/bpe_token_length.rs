use std::path::PathBuf;

use clap::Parser;
use text_correction_utils::tokenization::{
    tokenizer, BPETokenizerConfig, SpecialConfig, TokenizeConfig, TokenizerConfig,
};

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long)]
    file: PathBuf,
}

fn main() {
    let args = Args::parse();

    let special = SpecialConfig::default();
    let tok = tokenizer(TokenizerConfig {
        special: special.clone(),
        language: None,
        tokenize: TokenizeConfig::BPE(BPETokenizerConfig {
            merge_file: args.file,
            use_graphemes: true,
            max_vocab_size: None,
        }),
    })
    .unwrap();

    let n = tok.vocab_size() - special.tokens.len();
    let mut total_bytes = Vec::new();
    for i in 0..n {
        let bytes = tok.de_tokenize(&[i as u32], true);
        total_bytes.push(bytes.len());
    }
    total_bytes.sort();
    let sum = total_bytes.iter().sum::<usize>();
    println!(
        "Total bytes: {}, Avg: {:.2}, Median: {}",
        sum,
        sum as f64 / n as f64,
        total_bytes[n / 2]
    );
}
