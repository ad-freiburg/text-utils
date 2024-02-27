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

    let parser = LR1GrammarParser::from_file(args.parser, args.lexer)?;
    let input = read_to_string(args.input)?;
    let parse = parser.parse(&input, false).ok_or_else(|| "Parse error")?;
    println!("{}", parse.pretty(&input, true));
    Ok(())
}
