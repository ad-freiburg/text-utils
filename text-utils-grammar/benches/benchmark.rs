use std::fs::{self, read_to_string};
use std::path::PathBuf;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use text_utils_grammar::{
    Constraint, LR1GrammarConstraint, LR1GrammarParser, RegularExpressionConstraint,
};

fn load_continuations() -> Vec<Vec<u8>> {
    let dir = env!("CARGO_MANIFEST_DIR");
    let continuations_json = fs::read(PathBuf::from(dir).join("resources/test/continuations.json"))
        .expect("failed to read file");
    // use serde to deserialize continuations array from json
    serde_json::from_slice::<Vec<String>>(&continuations_json)
        .unwrap()
        .into_iter()
        .map(|c| c.as_bytes().to_vec())
        .collect()
}

fn bench_re_constraint(c: &mut Criterion) {
    let conts = load_continuations();

    let re = RegularExpressionConstraint::new(r"yes|no|maybe", conts.clone()).unwrap();
    let state = re.get_state(b"may").unwrap();
    c.bench_function("re_mc_get_valid_continuations", |b| {
        b.iter(|| re.get_valid_continuations_with_state(&state))
    });
    c.bench_function("re_mc_get_valid_continuations_with_states", |b| {
        b.iter_batched(
            || [state; 10],
            |input| re.get_valid_continuations_with_states(&input),
            BatchSize::SmallInput,
        )
    });

    let re = RegularExpressionConstraint::new(r"\w+@\w+\.(com|de|org)", conts.clone()).unwrap();
    let state = re.get_state(b"test").unwrap();
    c.bench_function("re_email1_get_valid_continuations", |b| {
        b.iter(|| re.get_valid_continuations_with_state(&state))
    });
    let state = re.get_state(b"test@gmai").unwrap();
    c.bench_function("re_email2_get_valid_continuations", |b| {
        b.iter(|| re.get_valid_continuations_with_state(&state))
    });
    let state = re.get_state(b"test@gmail.c").unwrap();
    c.bench_function("re_email3_get_valid_continuations", |b| {
        b.iter(|| re.get_valid_continuations_with_state(&state))
    });

    let dir = env!("CARGO_MANIFEST_DIR");
    let files = ["json.txt", "template.txt", "rdf_triples.txt"];
    let prefixes = [
        r#"
{
    "name": "irableirny",
    "age": "60",
    "email": "strvir"#,
        r#"
<name>irableirny</name>
<age>60</age>
<email>strvir"#,
        r#"
<bos>obiernobpb</eos>
<bop>aseimarbar</eop>
<boo>positorybo<eoo>
.
<bos>abilushcji</eos>
<bop>nomek"#,
    ];
    for (file, prefix) in files.iter().zip(prefixes) {
        let path = PathBuf::from(dir).join("resources/test/").join(file);
        let file_name = path.file_stem().unwrap().to_str().unwrap();
        let re = RegularExpressionConstraint::from_file(&path, conts.clone()).unwrap();
        let state = re.get_state(prefix.as_bytes()).unwrap();
        assert!(
            !re.get_valid_continuations_with_state(&state).0.is_empty(),
            "'{prefix}' has no valid continuations"
        );
        c.bench_function(
            &format!("re_file_{file_name}_get_valid_continuations"),
            |b| b.iter(|| re.get_valid_continuations_with_state(&state)),
        );
        c.bench_function(
            &format!("re_file_{file_name}_get_valid_continuations_with_states"),
            |b| {
                b.iter_batched(
                    || [state; 10],
                    |input| re.get_valid_continuations_with_states(&input),
                    BatchSize::SmallInput,
                )
            },
        );
    }
}

fn bench_lr1_constraint(c: &mut Criterion) {
    let conts = load_continuations();
    let grammar = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("grammars/json/json.y")
        .to_str()
        .unwrap()
        .to_string();
    let tokens = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("grammars/json/json.l")
        .to_str()
        .unwrap()
        .to_string();
    let lr1_constraint = LR1GrammarConstraint::from_file(grammar, tokens, conts.clone()).unwrap();
    let state = lr1_constraint.get_start_state();
    c.bench_function("lr1_json_empty_get_valid_continuations", |b| {
        b.iter(|| lr1_constraint.get_valid_continuations_with_state(&state))
    });
    let input = read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("grammars/json/examples/numbers.json"),
    )
    .unwrap();
    let state = lr1_constraint
        .get_state(&input.as_bytes()[..input.len() / 2])
        .unwrap();
    c.bench_function("lr1_json_numbers_get_valid_continuations", |b| {
        b.iter(|| lr1_constraint.get_valid_continuations_with_state(&state))
    });
    let input = read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("grammars/json/examples/example1.json"),
    )
    .unwrap();
    let state = lr1_constraint
        .get_state(&input.as_bytes()[..input.len() / 2])
        .unwrap();
    c.bench_function("lr1_json_example1_get_valid_continuations", |b| {
        b.iter(|| lr1_constraint.get_valid_continuations_with_state(&state))
    });
}

fn bench_lr1_parser(c: &mut Criterion) {
    let grammar = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("grammars/json/json.y")
        .to_str()
        .unwrap()
        .to_string();
    let tokens = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("grammars/json/json.l")
        .to_str()
        .unwrap()
        .to_string();
    let parser = LR1GrammarParser::from_file(grammar, tokens).unwrap();
    let input = read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("grammars/json/examples/numbers.json"),
    )
    .unwrap();
    c.bench_function("lr1_parse_numbers", |b| {
        b.iter(|| parser.parse(&input, false))
    });
    let input = read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("grammars/json/examples/example1.json"),
    )
    .unwrap();
    c.bench_function("lr1_parse_example1", |b| {
        b.iter(|| parser.parse(&input, false))
    });
}

criterion_group!(
    benches,
    bench_re_constraint,
    bench_lr1_constraint,
    bench_lr1_parser
);
criterion_main!(benches);
