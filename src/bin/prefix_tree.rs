use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use clap::Parser;
use text_correction_utils::{
    prefix::PrefixTreeSearch, prefix_tree::Node, text::file_size, utils::progress_bar,
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

    let mut trie = Node::default();

    println!(
        "trie node empty size: {} bytes",
        std::mem::size_of::<Node<String>>()
    );

    let file = File::open(&args.file)?;
    for (idx, line) in pbar.wrap_iter(BufReader::new(file).lines()).enumerate() {
        let line = line?;
        let splits: Vec<_> = line.split('\t').collect();
        trie.insert(splits[0].trim(), splits[1].trim().to_string());
        if (idx + 1) % (num_lines / 10) == 0 || idx + 1 == num_lines {
            println!("num elements: {}", trie.size());
        }
    }

    Ok(())
}
