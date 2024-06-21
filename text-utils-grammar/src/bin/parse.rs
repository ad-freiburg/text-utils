use std::error::Error;
use std::time::{Duration, Instant};

use clap::Parser;
use text_utils_grammar::LR1GrammarParser;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    lexer: String,

    parser: String,

    input: String,

    #[clap(long, short)]
    prefix: bool,

    #[clap(long, short, default_value = "1000")]
    n: u32,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let parser = LR1GrammarParser::from_files(args.parser, args.lexer)?;

    let mut elapsed = Duration::ZERO;
    for i in 0..args.n {
        let start = Instant::now();

        let parse = if args.prefix {
            parser.prefix_parse(args.input.as_bytes(), false, false)
        } else {
            parser.parse(&args.input, false, false)
        }
        .unwrap();
        let end = Instant::now();
        elapsed += end - start;
        if i == 0 {
            println!("{}", parse.pretty(&args.input, true, false));
        }
    }
    println!(
        "Elapsed: {:.2}ms",
        (elapsed / args.n).as_micros() as f64 / 1000.0
    );
    Ok(())
}
