use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use clap::Parser;
use text_correction_utils::{
    prefix::PrefixTreeSearch, prefix_vec::PrefixVec, text::file_size, utils::progress_bar,
};

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long)]
    file: String,

    #[arg(short, long)]
    progress: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let (num_lines, _) = file_size(&args.file)?;

    let pbar = progress_bar("reading file", num_lines as u64, !args.progress);

    println!(
        "prefix vec empty size: {} bytes",
        std::mem::size_of::<PrefixVec<u64>>()
    );

    let file = File::open(&args.file)?;
    let pfx: PrefixVec<_> = pbar
        .wrap_iter(BufReader::new(file).lines().filter_map(|line| {
            if let Ok(line) = line {
                let splits: Vec<_> = line.split('\t').collect();
                Some((
                    splits[0].trim().as_bytes().to_vec(),
                    splits[1]
                        .trim()
                        .chars()
                        .skip(1)
                        .collect::<String>()
                        .parse::<u64>()
                        .unwrap(),
                ))
            } else {
                None
            }
        }))
        .collect();

    println!("num elements: {}", pfx.size());

    Ok(())
}
