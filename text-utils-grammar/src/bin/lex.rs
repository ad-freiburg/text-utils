use std::time::{Duration, Instant};
use std::{error::Error, fs::read_to_string};

use clap::Parser;
use text_utils_grammar::LR1GrammarParser;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    lexer: String,

    parser: String,

    input: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let parser = LR1GrammarParser::from_files(args.parser, args.lexer)?;
    let input = read_to_string(args.input)?;

    let mut elapsed = Duration::ZERO;
    let n = 1000;
    for i in 0..n {
        let start = Instant::now();
        let lex = parser.lex(&input)?;
        let end = Instant::now();
        elapsed += end - start;
        if i == 0 {
            println!("tokens: {:?}", lex);
        }
    }
    println!(
        "Elapsed: {:.2}ms",
        (elapsed / n).as_micros() as f64 / 1000.0
    );
    Ok(())
}
