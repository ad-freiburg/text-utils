use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    error::Error,
    fmt::Display,
    fs::File,
    io::read_to_string,
    path::Path,
};

use cfgrammar::{
    yacc::{YaccGrammar, YaccKind, YaccOriginalActionKind},
    NewlineCache, TIdx,
};
use lrlex::{DefaultLexeme, DefaultLexerTypes, LRNonStreamingLexer};
use lrpar::{Lexeme, Node, RTParserBuilder};
use lrtable::{Minimiser, StIdx, StateGraph, StateTable};
use regex::Regex;
use regex_automata::util::primitives::StateID;

use crate::{
    utils::{PrefixDFA, PrefixMatch},
    Constraint,
};

fn load_grammar_and_pdfas(
    grammar: &str,
    grammar_kind: YaccKind,
    tokens: &str,
) -> Result<(YaccGrammar, Vec<(PrefixDFA, Option<TIdx<u32>>)>), Box<dyn Error>> {
    let grammar = YaccGrammar::new(grammar_kind, grammar)
        .map_err(|e| format!("errors creating grammar: {:#?}", e))?;

    // get token patterns and corresponding pdfas
    let token_regex = Regex::new("(?m)^(.+)\\s+(?:\"(.+)\"|(;))\\s*$")?;
    let sep = Regex::new("(?m)^%%$")?;
    let m = sep.find(tokens).ok_or("line with %% not found")?;
    let tokens = &tokens[m.end()..];

    // preprocess token names
    let mut token_names = HashMap::new();
    for tidx in grammar.iter_tidxs() {
        let Some(name) = grammar.token_name(tidx) else {
                if tidx != grammar.eof_token_idx() {
                    return Err(format!("no name found for token {:?}", tidx).into());
                }
                continue;
            };
        token_names.insert(name, tidx);
    }

    // build pdfas
    let mut pdfas = vec![];
    let mut seen = HashSet::new();
    for cap in token_regex.captures_iter(tokens) {
        let pattern = cap.get(1).unwrap().as_str();
        let pdfa = PrefixDFA::new(pattern)?;
        if pdfa.is_match_state(pdfa.get_start_state()) {
            return Err(format!("token pattern matches empty string: '{pattern}'").into());
        };
        if cap.get(3).is_some() {
            pdfas.push((pdfa, None));
            continue;
        }
        let name = cap.get(2).unwrap().as_str();
        let Some(&tidx) = token_names.get(name) else {
                continue;
            };
        if !seen.insert(name) {
            return Err(format!("duplicate token name: '{name}'").into());
        }
        pdfas.push((pdfa, Some(tidx)));
    }

    Ok((grammar, pdfas))
}

pub type Span = (usize, usize);

fn prefix_lexer(
    prefix: &[u8],
    pdfas: &[(PrefixDFA, Option<TIdx<u32>>)],
) -> Option<(Vec<(TIdx<u32>, Span)>, Vec<(usize, StateID, Span)>)> {
    // returns a list of tokens and a list of indices of pdfas matching
    // the rest of the prefix, or None if no matching pdfa is found
    let mut tokens = vec![];
    let mut prefix_matches = vec![];
    let mut i = 0;
    // logic is that longest match wins
    while i < prefix.len() {
        let mut longest = 0;
        let mut matching = None;
        for (pidx, (pdfa, tidx)) in pdfas.iter().enumerate() {
            match pdfa.find_prefix_match(pdfa.get_start_state(), &prefix[i..]) {
                PrefixMatch::None => continue,
                PrefixMatch::Maybe(state) => {
                    prefix_matches.push((pidx, state, (i, prefix.len() - i)))
                }
                PrefixMatch::UpTo(end, _) => {
                    if end > longest {
                        longest = end;
                        matching = tidx.map(|tidx| (tidx, (i, end)));
                    }
                }
            };
        }
        if !prefix_matches.is_empty() {
            // there is at least one pdfa that matches the whole rest of prefix
            break;
        } else if longest > 0 {
            if let Some(tidx) = matching {
                tokens.push(tidx);
            }
            i += longest;
        } else {
            return None;
        }
    }
    Some((tokens, prefix_matches))
}

fn lexer(text: &str, pdfas: &[(PrefixDFA, Option<TIdx<u32>>)]) -> Option<Vec<(TIdx<u32>, Span)>> {
    let Some((mut tokens, last_matches)) = prefix_lexer(text.as_bytes(), pdfas) else {
        return None;
    };
    if let Some(item) = last_matches.iter().find_map(|&(pidx, state, span)| {
        let (pdfa, tidx) = &pdfas[pidx];
        if pdfa.is_match_state(state) {
            tidx.map(|tidx| (tidx, span))
        } else {
            None
        }
    }) {
        tokens.push(item);
    }
    Some(tokens)
}

pub struct LR1GrammarParser {
    grammar: YaccGrammar<u32>,
    table: StateTable<u32>,
    pdfas: Vec<(PrefixDFA, Option<TIdx<u32>>)>,
}

#[derive(Debug, PartialEq)]
pub enum LR1Parse<'a> {
    Terminal(&'a str, Span),
    NonTerminal(&'a str, Vec<LR1Parse<'a>>),
}

impl LR1Parse<'_> {
    pub fn pretty(&self, text: &str, collapse: bool) -> String {
        fn pretty_parse(parse: &LR1Parse<'_>, indent: usize, text: &str, collapse: bool) -> String {
            match parse {
                LR1Parse::Terminal(name, span) => {
                    let &(start, len) = span;
                    format!("{:indent$}{}: '{}'", "", name, &text[start..start + len],)
                }
                LR1Parse::NonTerminal(name, children) => {
                    assert!(!children.is_empty());
                    if children.len() == 1 && collapse {
                        return pretty_parse(&children[0], indent, text, collapse);
                    }
                    let mut s = format!("{:indent$}{}", "", name);
                    for child in children {
                        s.push('\n');
                        s.push_str(&pretty_parse(child, indent + 2, text, collapse));
                    }
                    s
                }
            }
        }
        pretty_parse(self, 0, text, collapse)
    }
}

impl LR1GrammarParser {
    pub fn new(grammar: &str, tokens: &str) -> Result<Self, Box<dyn Error>> {
        let (grammar, pdfas) = load_grammar_and_pdfas(
            grammar,
            YaccKind::Original(YaccOriginalActionKind::GenericParseTree),
            tokens,
        )?;
        let (_, table) = lrtable::from_yacc(&grammar, Minimiser::Pager)?;
        Ok(Self {
            grammar,
            table,
            pdfas,
        })
    }

    pub fn from_file(
        grammar_path: impl AsRef<Path>,
        tokens_path: impl AsRef<Path>,
    ) -> Result<Self, Box<dyn Error>> {
        let file = File::open(grammar_path.as_ref())?;
        let grammar = read_to_string(file)?;
        let file = File::open(tokens_path.as_ref())?;
        let tokens = read_to_string(file)?;
        Self::new(&grammar, &tokens)
    }

    pub fn parse(&self, text: &str, collapse: bool) -> Option<LR1Parse<'_>> {
        let lexer = LRNonStreamingLexer::new(
            text,
            lexer(text, &self.pdfas)?
                .into_iter()
                .map(|(tidx, (start, len))| Ok(DefaultLexeme::new(tidx.as_storaget(), start, len)))
                .collect(),
            NewlineCache::new(),
        );
        let parser: RTParserBuilder<'_, u32, DefaultLexerTypes> =
            RTParserBuilder::new(&self.grammar, &self.table);
        let (tree, errors) = parser.parse_generictree(&lexer);
        if !errors.is_empty() {
            return None;
        }
        let tree = tree?;
        // convert tree to lr1 parse
        fn node_to_lr1<'a>(
            grammar: &'a YaccGrammar,
            node: &Node<DefaultLexeme<u32>, u32>,
            collapse: bool,
        ) -> Option<LR1Parse<'a>> {
            match node {
                Node::Term { lexeme } => {
                    let span = lexeme.span();
                    let tidx = lexeme.tok_id();
                    let tname = grammar.token_name(TIdx(tidx))?;
                    Some(LR1Parse::Terminal(tname, (span.start(), span.len())))
                }
                Node::Nonterm { ridx, nodes } => {
                    assert!(!nodes.is_empty());
                    if nodes.len() == 1 && collapse {
                        return node_to_lr1(grammar, &nodes[0], collapse);
                    }
                    let rname = grammar.rule_name_str(*ridx);
                    Some(LR1Parse::NonTerminal(
                        rname,
                        nodes
                            .iter()
                            .map(|node| node_to_lr1(grammar, node, collapse))
                            .collect::<Option<_>>()?,
                    ))
                }
            }
        }
        node_to_lr1(&self.grammar, &tree, collapse)
    }
}

pub struct LR1GrammarConstraint {
    pub(crate) grammar: YaccGrammar<u32>,
    graph: StateGraph<u32>,
    table: StateTable<u32>,
    pdfas: Vec<(PrefixDFA, Option<TIdx<u32>>)>,
    continuations: Vec<Vec<u8>>,
}

impl LR1GrammarConstraint {
    pub fn new(
        grammar: &str,
        tokens: &str,
        continuations: Vec<Vec<u8>>,
    ) -> Result<Self, Box<dyn Error>> {
        let (grammar, pdfas) = load_grammar_and_pdfas(
            grammar,
            YaccKind::Original(YaccOriginalActionKind::NoAction),
            tokens,
        )?;
        let (graph, table) = lrtable::from_yacc(&grammar, Minimiser::Pager)?;

        Ok(Self {
            continuations,
            grammar,
            graph,
            pdfas,
            table,
        })
    }

    pub fn from_file(
        grammar_path: impl AsRef<Path>,
        tokens_path: impl AsRef<Path>,
        continuations: Vec<Vec<u8>>,
    ) -> Result<Self, Box<dyn Error>> {
        let file = File::open(grammar_path.as_ref())?;
        let grammar = read_to_string(file)?;
        let file = File::open(tokens_path.as_ref())?;
        let tokens = read_to_string(file)?;
        Self::new(&grammar, &tokens, continuations)
    }
}

pub struct LR1State {
    stack: Vec<StIdx<u32>>,
    matching: Vec<(usize, StateID)>,
}

impl Constraint for LR1GrammarConstraint {
    type State = LR1State;

    fn is_match_state(&self, state: Self::State) -> bool {
        let Some(&stidx) = state.stack.last() else {
            return false;
        };
        true
        // state.token_pdfas.iter().any(|&(tidx, pdfa_state)| {
        //     let Some(pdfa) = self.token_pdfas.get(&tidx) else {
        //         return false;
        //     };
        //     pdfa.is_match_state(pdfa_state)
        //         && matches!(self.table.action(stidx, tidx), Action::Accept)
        // })
    }

    fn get_valid_continuations_with_state(
        &self,
        state: Self::State,
    ) -> (Vec<usize>, Vec<Self::State>) {
        let Some(&stidx) = state.stack.last() else {
            return (vec![], vec![]);
        };
        //
        let mut cont_indices: BTreeSet<usize> = BTreeSet::new();
        let mut next_states: BTreeMap<usize, Self::State> = BTreeMap::new();
        // for (tidx, state) in state.token_pdfas {
        //     let pdfa = &self.token_pdfas[&tidx];
        //     for (cidx, next_state) in
        //         pdfa.valid_continuations_and_states(state, &self.continuations)
        //     {
        //         cont_indices.insert(cidx);
        //         next_states.insert(
        //             cidx,
        //             Self::State {
        //                 stack: state.stack.clone(),
        //                 token_pdfas: state
        //                     .token_pdfas
        //                     .iter()
        //                     .map(|(tidx, state)| (*tidx, *state))
        //                     .collect(),
        //                 other_pdfas: vec![],
        //             },
        //         );
        //     }
        // }
        // for (pidx, state) in state.other_pdfas {
        //     let pdfa = &self.other_pdfas[pidx];
        //     for (cidx, next_state) in
        //         pdfa.valid_continuations_and_states(state, &self.continuations)
        //     {
        //         cont_indices.insert(cidx);
        //     }
        // }
        assert_eq!(cont_indices.len(), next_states.len());
        (
            cont_indices.into_iter().collect(),
            next_states.into_values().collect(),
        )
    }

    fn get_start_state(&self) -> Self::State {
        let state = self.table.start_state();
        todo!();
        // Self::State {
        //     stack: vec![state],
        //     token_pdfas: self
        //         .table
        //         .state_actions(state)
        //         .filter_map(|tidx| {
        //             let pdfa = self.token_pdfas.get(&tidx)?;
        //             Some((tidx, pdfa.get_start_state()))
        //         })
        //         .collect(),
        //     other_pdfas: self
        //         .other_pdfas
        //         .iter()
        //         .map(|pdfa| pdfa.get_start_state())
        //         .enumerate()
        //         .collect(),
        // }
    }

    fn get_state(&self, _: &[u8]) -> Option<Self::State> {
        // fix this by parsing prefix into tokens with the lexer
        // and then driving the pda with these tokens
        unimplemented!(
            "getting state with a specific prefix is currently not supported for LR1 grammars"
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::{fs, path::PathBuf};

    fn load_continuations() -> Vec<Vec<u8>> {
        let dir = env!("CARGO_MANIFEST_DIR");
        let continuations_json =
            fs::read(PathBuf::from(dir).join("resources/test/continuations.json"))
                .expect("failed to read file");

        // use serde to deserialize continuations array from json
        serde_json::from_slice::<Vec<String>>(&continuations_json)
            .unwrap()
            .into_iter()
            .map(|c| c.as_bytes().to_vec())
            .collect()
    }

    fn get_calc_pfdas() -> (
        Vec<(PrefixDFA, Option<TIdx<u32>>)>,
        HashMap<TIdx<u32>, &'static str>,
    ) {
        // this simulates the pdfas we would get from a calc.l file
        (
            vec![
                (PrefixDFA::new("\\(").unwrap(), Some(TIdx(0))),
                (PrefixDFA::new("\\)").unwrap(), Some(TIdx(1))),
                (PrefixDFA::new("\\+").unwrap(), Some(TIdx(2))),
                (PrefixDFA::new("\\*").unwrap(), Some(TIdx(3))),
                (PrefixDFA::new("[0-9]+").unwrap(), Some(TIdx(4))),
                (PrefixDFA::new("[ ]+").unwrap(), None),
                (PrefixDFA::new("[\n\t]+").unwrap(), None),
            ],
            HashMap::from([
                (TIdx(0), "LP"),
                (TIdx(1), "RP"),
                (TIdx(2), "PLUS"),
                (TIdx(3), "TIMES"),
                (TIdx(4), "INT"),
            ]),
        )
    }

    fn get_ab_pdfas() -> (
        Vec<(PrefixDFA, Option<TIdx<u32>>)>,
        HashMap<TIdx<u32>, &'static str>,
    ) {
        (
            vec![
                (PrefixDFA::new("a").unwrap(), Some(TIdx(0))),
                (PrefixDFA::new("aa").unwrap(), Some(TIdx(1))),
                (PrefixDFA::new("ab").unwrap(), Some(TIdx(2))),
                (PrefixDFA::new("b").unwrap(), Some(TIdx(3))),
                (PrefixDFA::new("bb").unwrap(), Some(TIdx(4))),
                (PrefixDFA::new("ab").unwrap(), Some(TIdx(5))),
                (PrefixDFA::new("[ ]+").unwrap(), None),
                (PrefixDFA::new("[\n\t]+").unwrap(), None),
            ],
            HashMap::from([
                (TIdx(0), "A"),
                (TIdx(1), "AA"),
                (TIdx(2), "AB1"),
                (TIdx(3), "B"),
                (TIdx(4), "BB"),
                (TIdx(5), "AB2"),
            ]),
        )
    }

    #[test]
    fn test_lexer() {
        let (pdfas, map) = get_calc_pfdas();
        assert!(lexer("2 - 1", &pdfas).is_none());
        let lexemes = lexer("(1 + 28)*\n3", &pdfas).unwrap();
        let lexemes: Vec<_> = lexemes
            .into_iter()
            .map(|(tidx, (start, len))| (map[&tidx], (start, len)))
            .collect();
        assert_eq!(
            lexemes,
            vec![
                ("LP", (0, 1)),
                ("INT", (1, 1)),
                ("PLUS", (3, 1)),
                ("INT", (5, 2)),
                ("RP", (7, 1)),
                ("TIMES", (8, 1)),
                ("INT", (10, 1))
            ]
        );
        let (pdfas, map) = get_ab_pdfas();
        let lexemes = lexer("aabb", &pdfas).unwrap();
        let lexemes: Vec<_> = lexemes
            .into_iter()
            .map(|(tidx, (start, len))| (map[&tidx], (start, len)))
            .collect();
        assert_eq!(lexemes, vec![("AA", (0, 2)), ("BB", (2, 2))]);
        let lexemes = lexer("abb", &pdfas).unwrap();
        let lexemes: Vec<_> = lexemes
            .into_iter()
            .map(|(tidx, (start, len))| (map[&tidx], (start, len)))
            .collect();
        assert_eq!(lexemes, vec![("AB1", (0, 2)), ("B", (2, 1))]);
        assert!(lexer("abac", &pdfas).is_none());
    }

    #[test]
    fn test_prefix_lexer() {
        let (pdfas, map) = get_calc_pfdas();
        let (lexemes, matching) = prefix_lexer(b"(1 + 28)*\n3", &pdfas).unwrap();
        let lexemes: Vec<_> = lexemes
            .into_iter()
            .map(|(tidx, (start, len))| (map[&tidx], (start, len)))
            .collect();
        assert_eq!(
            lexemes,
            vec![
                ("LP", (0, 1)),
                ("INT", (1, 1)),
                ("PLUS", (3, 1)),
                ("INT", (5, 2)),
                ("RP", (7, 1)),
                ("TIMES", (8, 1)),
            ]
        );
        assert_eq!(matching.len(), 1);
        let (idx, state, span) = matching[0];
        assert_eq!(idx, 4);
        let (pdfa, tidx) = &pdfas[idx];
        assert_eq!(map[tidx.as_ref().unwrap()], "INT");
        assert!(pdfa.is_match_state(state));
        assert_eq!(span, (10, 1));
        let (lexemes, matching) = prefix_lexer(b"", &pdfas).unwrap();
        assert!(lexemes.is_empty());
        assert!(matching.is_empty());
        let (lexemes, matching) = prefix_lexer(b"    (", &pdfas).unwrap();
        assert!(lexemes.is_empty());
        assert_eq!(matching.len(), 1);
        let (idx, state, span) = matching[0];
        assert_eq!(idx, 0);
        let (pdfa, tidx) = &pdfas[idx];
        assert_eq!(map[tidx.as_ref().unwrap()], "LP");
        assert!(pdfa.is_match_state(state));
        assert_eq!(span, (4, 1));

        let (pdfas, map) = get_ab_pdfas();
        let (lexemes, matching) = prefix_lexer(b"aabb", &pdfas).unwrap();
        let lexemes: Vec<_> = lexemes
            .into_iter()
            .map(|(tidx, (start, len))| (map[&tidx], (start, len)))
            .collect();
        assert_eq!(lexemes, vec![("AA", (0, 2))]);
        assert_eq!(matching.len(), 1);
        let (idx, state, span) = matching[0];
        assert_eq!(idx, 4);
        let (pdfa, tidx) = &pdfas[idx];
        assert_eq!(map[tidx.as_ref().unwrap()], "BB");
        assert!(pdfa.is_match_state(state));
        assert_eq!(span, (2, 2));

        let (lexemes, matching) = prefix_lexer(b"aab", &pdfas).unwrap();
        let lexemes: Vec<_> = lexemes
            .into_iter()
            .map(|(tidx, (start, len))| (map[&tidx], (start, len)))
            .collect();
        assert_eq!(lexemes, vec![("AA", (0, 2))]);
        assert_eq!(matching.len(), 2);
        let (idx, state, span) = matching[0];
        assert_eq!(idx, 3);
        let (pdfa, tidx) = &pdfas[idx];
        assert_eq!(map[tidx.as_ref().unwrap()], "B");
        assert!(pdfa.is_match_state(state));
        assert_eq!(span, (2, 1));
        let (idx, state, span) = matching[1];
        assert_eq!(idx, 4);
        let (pdfa, tidx) = &pdfas[idx];
        assert_eq!(map[tidx.as_ref().unwrap()], "BB");
        assert!(!pdfa.is_match_state(state));
        assert_eq!(span, (2, 1));
    }

    #[test]
    fn test_lrk_parser() {
        let grammar =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("resources/test/grammars/calc.y");
        let tokens =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("resources/test/grammars/calc.l");
        let lrk = LR1GrammarParser::from_file(grammar, tokens).unwrap();
        assert_eq!(lrk.parse("2 - 1", false), None);
        let text = "(1 + 28)*\n3";
        let parse = lrk.parse(text, true).unwrap();
        println!("{}", parse.pretty(text, true));
    }

    #[test]
    fn test_lrk_constraint() {
        let grammar =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("resources/test/grammars/calc.y");
        let tokens =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("resources/test/grammars/calc.l");
        let conts = load_continuations();
        let lrk = LR1GrammarConstraint::from_file(grammar, tokens, conts.clone()).unwrap();
        let state = lrk.get_start_state();
        let (cont_indices, _) = lrk.get_valid_continuations_with_state(state);
        println!(
            "{:?}",
            cont_indices
                .iter()
                .map(|i| String::from_utf8_lossy(&conts[*i]))
                .collect::<Vec<_>>()
        );
        let lb = lrk.grammar.token_idx("LP").unwrap();
        let int = lrk.grammar.token_idx("INT").unwrap();
        // let state = lrk.drive_with_tokens(&[lb, int]).unwrap();
        let state = lrk.get_start_state();
        let (cont_indices, _) = lrk.get_valid_continuations_with_state(state);
        println!(
            "{:?}",
            cont_indices
                .iter()
                .map(|i| String::from_utf8_lossy(&conts[*i]))
                .collect::<Vec<_>>()
        );
        let grammar =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("resources/test/grammars/test.y");
        let tokens =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("resources/test/grammars/test.l");
        let lrk = LR1GrammarConstraint::from_file(grammar, tokens, conts.clone()).unwrap();
        let state = lrk.get_start_state();
        let (cont_indices, _) = lrk.get_valid_continuations_with_state(state);
        println!(
            "{:?}",
            cont_indices
                .iter()
                .map(|i| String::from_utf8_lossy(&conts[*i]))
                .collect::<Vec<_>>()
        );
    }
}
