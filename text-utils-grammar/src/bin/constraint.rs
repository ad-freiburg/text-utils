use std::error::Error;
use std::fs::read_to_string;
use std::time::{Duration, Instant};

use clap::Parser;
use text_utils_grammar::{Constraint, LR1GrammarConstraint};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    lexer: String,

    parser: String,

    continuations: String,

    input: String,

    #[clap(long, short, default_value = "1000")]
    n: u32,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let continuations_json = read_to_string(args.continuations)?;
    // use serde to deserialize continuations array from json
    let continuations = serde_json::from_slice::<Vec<String>>(continuations_json.as_bytes())
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|c| c.as_bytes().to_vec())
        .collect();

    let constraint = LR1GrammarConstraint::from_files(args.parser, args.lexer, continuations)?;
    let state = constraint
        .get_state(args.input.as_bytes())
        .ok_or("failed to get state")?;
    println!("state: {:?}", state);

    let mut elapsed = Duration::ZERO;
    for i in 0..args.n {
        let start = Instant::now();

        let conts = constraint.get_valid_continuations(&state);
        if i == 0 {
            println!("num const: {}", conts.len());
        }

        let end = Instant::now();
        elapsed += end - start;
    }
    println!(
        "Elapsed: {:.2}ms",
        (elapsed / args.n).as_micros() as f64 / 1000.0
    );
    Ok(())
}
