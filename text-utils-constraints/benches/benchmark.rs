use std::fs;
use std::path::PathBuf;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use text_utils_constraints::{Constraint, RegularExpressionConstraint};

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

fn bench_re(c: &mut Criterion) {
    let conts = load_continuations();

    let re = RegularExpressionConstraint::new(r"yes|no|maybe", &conts).unwrap();
    let state = re.get_state(b"may");
    c.bench_function("re_mc_get_valid_continuations", |b| {
        b.iter(|| re.get_valid_continuations_with_state(state))
    });
    c.bench_function("re_mc_get_valid_continuations_with_prefix", |b| {
        b.iter(|| re.get_valid_continuations_with_prefix(b"may"))
    });
    let prefixes: Vec<Vec<u8>> = [b"may"; 10].into_iter().map(|s| s.to_vec()).collect();
    c.bench_function("re_mc_get_valid_continuations_with_prefixes", |b| {
        b.iter(|| re.get_valid_continuations_with_prefixes(&prefixes))
    });
    c.bench_function("re_mc_get_valid_continuations_with_states", |b| {
        b.iter_batched(
            || vec![state; 10],
            |input| re.get_valid_continuations_with_states(input),
            BatchSize::SmallInput,
        )
    });

    let re = RegularExpressionConstraint::new(r"\w+@\w+\.(com|de|org)", &conts).unwrap();
    let state = re.get_state(b"test");
    c.bench_function("re_email1_get_valid_continuations", |b| {
        b.iter(|| re.get_valid_continuations_with_state(state))
    });
    let state = re.get_state(b"test@gmai");
    c.bench_function("re_email2_get_valid_continuations", |b| {
        b.iter(|| re.get_valid_continuations_with_state(state))
    });
    let state = re.get_state(b"test@gmail.c");
    c.bench_function("re_email3_get_valid_continuations", |b| {
        b.iter(|| re.get_valid_continuations_with_state(state))
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
        let re = RegularExpressionConstraint::from_file(&path, &conts).unwrap();
        let state = re.get_state(prefix.as_bytes());
        assert!(
            !re.get_valid_continuations_with_prefix(prefix.as_bytes())
                .0
                .is_empty(),
            "'{prefix}' has no valid continuations"
        );
        c.bench_function(
            &format!("re_file_{file_name}_get_valid_continuations"),
            |b| b.iter(|| re.get_valid_continuations_with_state(state)),
        );
        c.bench_function(
            &format!("re_file_{file_name}_get_valid_continuations_with_prefix"),
            |b| b.iter(|| re.get_valid_continuations_with_prefix(prefix.as_bytes())),
        );
        let prefixes: Vec<_> = [prefix; 10]
            .into_iter()
            .map(|s| s.as_bytes().to_vec())
            .collect();
        c.bench_function(
            &format!("re_file_{file_name}_get_valid_continuations_with_prefixes"),
            |b| b.iter(|| re.get_valid_continuations_with_prefixes(&prefixes)),
        );
        c.bench_function(
            &format!("re_file_{file_name}_get_valid_continuations_with_states"),
            |b| {
                b.iter_batched(
                    || vec![state; 10],
                    |input| re.get_valid_continuations_with_states(input),
                    BatchSize::SmallInput,
                )
            },
        );
    }
}

criterion_group!(benches, bench_re);
criterion_main!(benches);
