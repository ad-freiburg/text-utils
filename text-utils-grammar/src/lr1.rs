use std::{collections::HashMap, error::Error, fs::File, io::read_to_string, path::Path};

use cfgrammar::{
    yacc::{YaccGrammar, YaccGrammarError, YaccKind, YaccOriginalActionKind},
    NewlineCache, Spanned, TIdx,
};
use indexmap::IndexMap;
use itertools::Itertools;
use lrlex::{DefaultLexeme, DefaultLexerTypes, LRNonStreamingLexer};
use lrpar::{Lexeme, Node, RTParserBuilder};
use lrtable::{Action, Minimiser, StIdx, StateTable};
use regex::{escape, Regex};
use regex_automata::util::primitives::StateID;

use crate::{
    utils::{optimized_prefix_order, PrefixDFA, PrefixMatch},
    Constraint,
};

enum Part {
    Literal(String),
    Regex(String),
}

// define function to recursively build pattern from parts
fn pattern_from_parts(
    name: &str,
    parts: &[Part],
    name_regex: &Regex,
    fragments: &HashMap<&str, &str>,
    tokens: &IndexMap<&str, Vec<Part>>,
) -> Result<String, Box<dyn Error>> {
    let mut pattern = String::new();
    for part in parts {
        match part {
            Part::Literal(s) => pattern.push_str(s),
            Part::Regex(s) => {
                // find all tokens or framents in regex
                // and replace them with their pattern
                let mut replaced = String::new();
                let mut last_match = 0;
                for caps in name_regex.captures_iter(s) {
                    let m = caps.get(0).unwrap();
                    replaced.push_str(&s[last_match..m.start()]);
                    // surround token or fragment with parentheses to group it
                    replaced.push_str("(?:");
                    let _name = caps.get(1).unwrap().as_str();
                    if let Some(parts) = tokens.get(_name) {
                        let replacement =
                            pattern_from_parts(name, parts, name_regex, fragments, tokens)?;
                        replaced.push_str(&replacement);
                    } else if let Some(pattern) = fragments.get(_name) {
                        let replacement = pattern_from_parts(
                            name,
                            &[Part::Regex(pattern.to_string())],
                            name_regex,
                            fragments,
                            tokens,
                        )?;
                        replaced.push_str(&replacement);
                    } else {
                        return Err(format!(
                            "token or fragment {_name} within {name} not found in lexer"
                        )
                        .into());
                    }
                    replaced.push(')');
                    last_match = m.end();
                }
                replaced.push_str(&s[last_match..]);
                pattern.push_str(&replaced);
            }
        }
    }
    Ok(pattern)
}

type PdfaList = Vec<(PrefixDFA, Option<TIdx<u32>>)>;

fn format_yacc_error(grammar: &str, e: &YaccGrammarError) -> String {
    format!(
        "{} at {}",
        e,
        e.spans()
            .iter()
            .map(|s| if s.is_empty() {
                let start = s.start().saturating_sub(20);
                let end = grammar.len().min(s.end() + 20);
                let context = &grammar.as_bytes()[start..end];
                format!("middle of '{}'", String::from_utf8_lossy(context))
            } else {
                format!("'{}'", &grammar[s.start()..s.end()])
            })
            .join(" and ")
    )
}

fn load_grammar_and_pdfas(
    grammar: &str,
    grammar_kind: YaccKind,
    lexer: &str,
) -> Result<(YaccGrammar, PdfaList), Box<dyn Error>> {
    let grammar = YaccGrammar::new(grammar_kind, grammar).map_err(|e| {
        format!(
            "errors creating grammar:\n{}",
            e.iter().map(|e| format_yacc_error(grammar, e)).join("\n")
        )
    })?;

    // get token patterns and corresponding pdfas
    let token_name = Regex::new(r"\{([A-Z0-9_]+)\}")?;
    let fragment_token_regex = Regex::new(r"(?m)^([A-Z0-9_]+|;)\s+(.+)$")?;
    let sep = Regex::new("(?m)^%%$")?;
    let m = sep.find(lexer).ok_or("line with %% not found")?;

    // parse fragements
    let mut fragments = HashMap::new();
    for line in lexer[..m.start()].lines() {
        if line.is_empty() || line.trim_start().starts_with("//") {
            continue;
        }
        let cap = fragment_token_regex
            .captures(line)
            .ok_or(format!("invalid fragment line: {line}"))?;
        let name = cap.get(1).unwrap().as_str();
        let pattern = cap.get(2).unwrap().as_str();
        if fragments.insert(name, pattern).is_some() {
            return Err(format!("duplicate fragment {name}").into());
        };
    }

    // parse tokens / terminals
    // use index map to preserve order
    let mut tokens = IndexMap::new();
    let mut ignore_tokens = vec![];
    for line in lexer[m.end()..].lines() {
        if line.is_empty() || line.trim_start().starts_with("//") {
            continue;
        }
        let cap = fragment_token_regex
            .captures(line)
            .ok_or(format!("invalid token line: {line}"))?;
        let name = cap.get(1).unwrap().as_str();
        let pattern = cap.get(2).unwrap().as_str();
        let mut parts = vec![];
        for part in pattern.split_whitespace() {
            if (part.starts_with('\'') && part.ends_with('\''))
                || (part.starts_with('"') && part.ends_with('"'))
            {
                // treat part as literal
                parts.push(Part::Literal(escape(&part[1..part.len() - 1])));
            } else {
                // treat part as regular expression
                parts.push(Part::Regex(part.to_string()));
            }
        }
        if parts.is_empty() {
            return Err(format!("invalid token pattern {pattern} for {name}").into());
        }
        if name == ";" {
            ignore_tokens.push(parts);
            continue;
        }
        if !ignore_tokens.is_empty() {
            return Err("ignore tokens must be at the end of the lexer file".into());
        }
        if grammar.token_idx(name).is_none() {
            eprintln!("token {name} not used in grammar, skipping...");
        };
        if tokens.insert(name, parts).is_some() {
            return Err(format!("duplicate token {name}").into());
        };
    }

    // build pdfas from fragments and tokens
    let mut pdfas = vec![];
    for (name, parts) in tokens.iter() {
        let pattern = pattern_from_parts(name, parts, &token_name, &fragments, &tokens)?;
        let pdfa = PrefixDFA::new(&pattern)?;
        if pdfa.is_match_state(pdfa.get_start_state()) {
            return Err(format!("token pattern {pattern} for {name} matches empty string").into());
        };
        pdfas.push((pdfa, grammar.token_idx(name)));
    }

    // add all unseen tokens from grammar as literal tokens to lexer
    for token in grammar
        .iter_tidxs()
        .filter_map(|tidx| grammar.token_name(tidx))
        .filter(|name| !fragments.contains_key(name) && !tokens.contains_key(name))
    {
        let tidx = grammar
            .token_idx(token)
            .ok_or(format!("token {token} not found in grammar"))?;
        let pdfa = PrefixDFA::new(&escape(token))?;
        pdfas.push((pdfa, Some(tidx)));
    }

    // add ignore pdfas at the end
    for parts in &ignore_tokens {
        let pattern = pattern_from_parts("ignore token", parts, &token_name, &fragments, &tokens)?;
        let pdfa = PrefixDFA::new(&pattern)?;
        pdfas.push((pdfa, None));
    }

    Ok((grammar, pdfas))
}

type Tokens = Vec<TIdx<u32>>;
type Span = (usize, usize);
type Spans = Vec<Span>;
type Matching = Vec<(usize, StateID)>;

fn prefix_lexer(
    prefix: &[u8],
    pdfas: &[(PrefixDFA, Option<TIdx<u32>>)],
) -> Result<(Tokens, Spans, Matching, Span), Box<dyn Error>> {
    // returns a list of tokens and a list of indices of pdfas matching
    // the rest of the prefix, or None if no matching pdfa is found
    let mut tokens = vec![];
    let mut spans = vec![];
    // initially all pdfas are in the matching state, the start state
    let mut prefix_matches: Vec<_> = pdfas
        .iter()
        .enumerate()
        .map(|(pidx, (pdfa, _))| (pidx, pdfa.get_start_state()))
        .collect();
    let mut i = 0;
    // logic is that longest match wins
    while i < prefix.len() {
        let mut longest = 0;
        let mut matching = None;
        prefix_matches.clear();
        for (pidx, (pdfa, tidx)) in pdfas.iter().enumerate() {
            match pdfa.find_prefix_match(pdfa.get_start_state(), &prefix[i..]) {
                PrefixMatch::None => continue,
                PrefixMatch::Maybe(state) => prefix_matches.push((pidx, state)),
                PrefixMatch::UpTo(end, _) => {
                    if end > longest {
                        longest = end;
                        matching = tidx.as_ref();
                    }
                }
            };
        }
        if !prefix_matches.is_empty() {
            // there is at least one pdfa that matches the whole rest of prefix
            break;
        } else if longest > 0 {
            if let Some(&tidx) = matching {
                tokens.push(tidx);
                spans.push((i, longest));
            }
            i += longest;
        } else {
            return Err(format!(
                "no matching token found from position {i}: '{}'",
                String::from_utf8_lossy(&prefix[i..])
            )
            .into());
        }
    }
    Ok((tokens, spans, prefix_matches, (i, prefix.len() - i)))
}

fn lexer(
    text: &str,
    pdfas: &[(PrefixDFA, Option<TIdx<u32>>)],
) -> Result<(Tokens, Spans), Box<dyn Error>> {
    let (mut tokens, mut spans, last_matches, last_span) = prefix_lexer(text.as_bytes(), pdfas)?;
    if let Some(&tidx) = last_matches.iter().find_map(|&(pidx, state)| {
        let (pdfa, Some(tidx)) = &pdfas[pidx] else {
            return None;
        };
        if pdfa.is_match_state(state) {
            Some(tidx)
        } else {
            None
        }
    }) {
        tokens.push(tidx);
        spans.push(last_span);
    }
    Ok((tokens, spans))
}

pub struct LR1GrammarParser {
    grammar: YaccGrammar<u32>,
    table: StateTable<u32>,
    pdfas: Vec<(PrefixDFA, Option<TIdx<u32>>)>,
}

#[derive(Debug, PartialEq)]
pub enum LR1Parse<'a> {
    Empty(&'a str),
    Terminal(&'a str, Span),
    NonTerminal(&'a str, Vec<LR1Parse<'a>>, Span),
}

impl LR1Parse<'_> {
    pub fn is_empty(&self) -> bool {
        matches!(self, LR1Parse::Empty(..))
    }

    pub fn span(&self) -> Option<&Span> {
        match self {
            LR1Parse::Empty(..) => None,
            LR1Parse::Terminal(.., span) => Some(span),
            LR1Parse::NonTerminal(.., span) => Some(span),
        }
    }

    pub fn pretty(&self, text: &str, collapse: bool) -> String {
        fn pretty_parse(parse: &LR1Parse<'_>, indent: usize, text: &str, collapse: bool) -> String {
            match parse {
                LR1Parse::Empty(name) => format!("{:indent$}{name}: %EMPTY%", ""),
                LR1Parse::Terminal(name, (start, len)) => {
                    format!("{:indent$}{name}: '{}'", "", &text[*start..*start + *len],)
                }
                LR1Parse::NonTerminal(name, children, ..) => {
                    assert!(!children.is_empty());
                    if children.len() == 1 && collapse {
                        return pretty_parse(&children[0], indent, text, collapse);
                    }
                    let mut s = format!("{:indent$}{name}", "");
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

    pub fn lex(&self, text: &str) -> Result<Vec<&str>, Box<dyn Error>> {
        let (tokens, _) = lexer(text, &self.pdfas)?;
        Ok(tokens
            .into_iter()
            .map(|tidx| self.grammar.token_name(tidx).unwrap())
            .collect())
    }

    pub fn parse(&self, text: &str, collapse: bool) -> Result<LR1Parse<'_>, Box<dyn Error>> {
        let (tokens, spans) = lexer(text, &self.pdfas)?;
        let lexer = LRNonStreamingLexer::new(
            text,
            tokens
                .into_iter()
                .zip(spans)
                .map(|(tidx, (start, len))| Ok(DefaultLexeme::new(tidx.as_storaget(), start, len)))
                .collect(),
            NewlineCache::new(),
        );
        let parser: RTParserBuilder<'_, u32, DefaultLexerTypes> =
            RTParserBuilder::new(&self.grammar, &self.table);
        let (tree, errors) = parser.parse_generictree(&lexer);
        if !errors.is_empty() {
            return Err(format!("errors parsing input:\n{}", errors.iter().join("\n")).into());
        }
        let Some(tree) = tree else {
            return Err("failed to parse input".into());
        };
        // convert tree to lr1 parse
        fn node_to_lr1<'a>(
            grammar: &'a YaccGrammar,
            node: &Node<DefaultLexeme<u32>, u32>,
            collapse: bool,
        ) -> LR1Parse<'a> {
            match node {
                Node::Term { lexeme } => {
                    let span = lexeme.span();
                    let tidx = lexeme.tok_id();
                    let tname = grammar.token_name(TIdx(tidx)).unwrap();
                    LR1Parse::Terminal(tname, (span.start(), span.len()))
                }
                Node::Nonterm { ridx, nodes } => {
                    let rname = grammar.rule_name_str(*ridx);
                    if nodes.is_empty() {
                        return LR1Parse::Empty(rname);
                    } else if nodes.len() == 1 && collapse {
                        return node_to_lr1(grammar, &nodes[0], collapse);
                    }
                    let nodes: Vec<_> = nodes
                        .iter()
                        .filter_map(|node| {
                            let node = node_to_lr1(grammar, node, collapse);
                            if node.is_empty() {
                                None
                            } else {
                                Some(node)
                            }
                        })
                        .collect();
                    if nodes.is_empty() {
                        return LR1Parse::Empty(rname);
                    }
                    let first_span = nodes.first().unwrap().span().unwrap();
                    let last_span = nodes.last().unwrap().span().unwrap();
                    let span = (first_span.0, last_span.0 + last_span.1 - first_span.0);
                    LR1Parse::NonTerminal(rname, nodes, span)
                }
            }
        }
        Ok(node_to_lr1(&self.grammar, &tree, collapse))
    }
}

pub struct LR1GrammarConstraint {
    pub(crate) grammar: YaccGrammar<u32>,
    table: StateTable<u32>,
    pdfas: Vec<(PrefixDFA, Option<TIdx<u32>>)>,
    continuations: Vec<Vec<u8>>,
    permutation: Vec<usize>,
    skips: Vec<usize>,
}

enum LR1Action {
    ShiftReduce(usize, StIdx<u32>),
    Accept,
    None,
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
        let (_, table) = lrtable::from_yacc(&grammar, Minimiser::Pager)?;
        let (permutation, skips) = optimized_prefix_order(&continuations);
        Ok(Self {
            continuations,
            grammar,
            pdfas,
            table,
            permutation,
            skips,
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

    fn shift_reduce(&self, stack: &[StIdx<u32>], token: TIdx<u32>) -> LR1Action {
        let Some(mut stidx) = stack.last().copied() else {
            return LR1Action::None;
        };
        // perform actions until the next shift,
        // can be implemented without actually
        // modifying the stack, because it will only ever
        // get smaller by reduces
        // stidx will always be the last element of the stack
        // (at position stack_end)
        let mut stack_end = stack.len() - 1;
        loop {
            match self.table.action(stidx, token) {
                Action::Shift(next_stidx) => {
                    stidx = next_stidx;
                    break;
                }
                Action::Reduce(pidx) => {
                    let ridx = self.grammar.prod_to_rule(pidx);
                    let rlen = self.grammar.prod(pidx).len();
                    stack_end -= rlen - 1;
                    let Some(new_stidx) = self.table.goto(stack[stack_end - 1], ridx) else {
                        return LR1Action::None;
                    };
                    stidx = new_stidx;
                }
                Action::Accept => return LR1Action::Accept,
                Action::Error => return LR1Action::None,
            };
        }
        LR1Action::ShiftReduce(stack_end + 1, stidx)
    }

    fn matching_pdfas(&self, stidx: StIdx<u32>) -> Vec<(usize, &PrefixDFA)> {
        let state_actions: Vec<_> = self.table.state_actions(stidx).collect();
        self.pdfas
            .iter()
            .enumerate()
            .filter_map(|(i, (pdfa, tidx))| {
                if tidx.is_none() || state_actions.contains(tidx.as_ref().unwrap()) {
                    Some((i, pdfa))
                } else {
                    None
                }
            })
            .collect()
    }
}

#[derive(Clone)]
pub struct LR1State {
    stack: Vec<StIdx<u32>>,
    matching: Matching,
}

impl LR1State {
    #[allow(dead_code)]
    pub fn next(&mut self, state: LR1NextState) {
        if let Some((keep, stidx, ..)) = state.action {
            self.stack.truncate(keep);
            self.stack.push(stidx);
        }
        self.matching = state.matching;
    }
}

#[derive(Clone, Default)]
pub struct LR1NextState {
    action: Option<(usize, StIdx<u32>, String)>,
    matching: Matching,
}

impl Constraint for LR1GrammarConstraint {
    type State = LR1State;
    type NextState = LR1NextState;

    fn is_match_state(&self, state: &Self::State) -> bool {
        state.matching.iter().any(|&(pidx, pdfa_state)| {
            let (pdfa, Some(token)) = &self.pdfas[pidx] else {
                return false;
            };
            if !pdfa.is_match_state(pdfa_state) {
                return false;
            }
            let LR1Action::ShiftReduce(keep, stidx) = self.shift_reduce(&state.stack, *token)
            else {
                return false;
            };
            let mut stack = state.stack[..keep].to_vec();
            stack.push(stidx);
            matches!(
                self.shift_reduce(&stack, self.grammar.eof_token_idx()),
                LR1Action::Accept
            )
        })
    }

    fn get_valid_continuations_with_state(
        &self,
        state: &Self::State,
    ) -> (Vec<usize>, Vec<Self::NextState>) {
        assert!(!state.matching.is_empty());
        let mut cont_indices = vec![];
        let mut next_states = vec![];

        // in case no pdfa is still matching for a continuation
        // we do the following:
        // 1. find all unskippable pdfas that are currently in matching state
        //    --> if there are none, skip
        // 2. select the one with the lowest index, as that would
        //    be the one picked by the lexer (length of match is the same
        //    for all pdfas)
        // 3. step with the corresponding token and return the action
        //    --> the action will later be used to create the next state
        let next = if let Some((LR1Action::ShiftReduce(keep, next_stidx), tidx)) = state
            .matching
            .iter()
            .find_map(|&(pidx, pdfa_state)| {
                let (pdfa, Some(tidx)) = &self.pdfas[pidx] else {
                    return None;
                };
                if pdfa.is_match_state(pdfa_state) {
                    Some(tidx)
                } else {
                    None
                }
            })
            .map(|&tidx| (self.shift_reduce(&state.stack, tidx), tidx))
        {
            let next_pdfas = self.matching_pdfas(next_stidx);
            let token_name = self.grammar.token_name(tidx).unwrap();
            Some(((keep, next_stidx, token_name.to_string()), next_pdfas))
        } else {
            None
        };

        let only_skippable_matching = state.matching.iter().all(|&(pidx, pdfa_state)| {
            let (pdfa, None) = &self.pdfas[pidx] else {
                return false;
            };
            pdfa.is_match_state(pdfa_state)
        });

        let matching_pdfas = self.matching_pdfas(*state.stack.last().unwrap());

        // now check all continuations
        let mut i = 0;
        while i < self.permutation.len() {
            let skip = self.skips[i];
            let j = self.permutation[i];
            let cont = &self.continuations[j];
            i += 1;

            // get all pdfas that are still matching
            let mut still_matching: Vec<_> = vec![];
            for &(pidx, pdfa_state) in &state.matching {
                let (pdfa, _) = &self.pdfas[pidx];
                if let Some(state) = pdfa.drive(pdfa_state, cont) {
                    still_matching.push((pidx, state));
                }
            }

            // if we have some pdfas that are still matching, use
            // them in the next state; this corresponds the
            // longest matching rule in the lexer
            if !still_matching.is_empty() {
                cont_indices.push(j);
                next_states.push(LR1NextState {
                    action: None,
                    matching: still_matching,
                });
            } else if only_skippable_matching {
                let matching: Vec<_> = matching_pdfas
                    .iter()
                    .filter_map(|&(i, pdfa)| {
                        pdfa.drive(pdfa.get_start_state(), cont)
                            .map(|state| (i, state))
                    })
                    .collect();

                if matching.is_empty() {
                    i += skip;
                    continue;
                }

                cont_indices.push(j);
                next_states.push(LR1NextState {
                    action: None,
                    matching,
                });
            } else if let Some((next_action, next_pdfas)) = &next {
                // if there are no matching pdfas, check the matching pdfas for the next state
                let next_matching: Vec<_> = next_pdfas
                    .iter()
                    .filter_map(|&(i, pdfa)| {
                        pdfa.drive(pdfa.get_start_state(), cont)
                            .map(|state| (i, state))
                    })
                    .collect();

                if next_matching.is_empty() {
                    i += skip;
                    continue;
                }

                cont_indices.push(j);
                next_states.push(LR1NextState {
                    action: Some(next_action.clone()),
                    matching: next_matching,
                });
            }
        }
        (cont_indices, next_states)
    }

    fn get_start_state(&self) -> Self::State {
        self.get_state(b"").expect("should not happen")
    }

    fn get_state(&self, prefix: &[u8]) -> Option<Self::State> {
        // fix this by parsing prefix into tokens with the lexer
        // and then driving the pda with these tokens
        let (tokens, _, matching, _) = prefix_lexer(prefix, &self.pdfas).ok()?;
        let mut stack = vec![self.table.start_state()];
        let mut idx = 0;
        while idx < tokens.len() {
            let stidx = stack.last()?;
            let tidx = tokens[idx];
            match self.table.action(*stidx, tidx) {
                Action::Shift(stidx) => {
                    stack.push(stidx);
                    idx += 1;
                }
                Action::Reduce(pidx) => {
                    let ridx = self.grammar.prod_to_rule(pidx);
                    let keep = stack.len() - self.grammar.prod(pidx).len();
                    stack.truncate(keep);
                    let stidx = self.table.goto(*stack.last()?, ridx)?;
                    stack.push(stidx);
                }
                Action::Accept | Action::Error => return None,
            }
        }
        Some(Self::State { stack, matching })
    }
}

#[cfg(test)]
mod test {
    use itertools::Itertools;

    use super::*;
    use std::{collections::HashMap, fs, path::PathBuf};

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
        let (tokens, spans) = lexer("(1 + 28)*\n3", &pdfas).unwrap();
        assert_eq!(
            tokens.into_iter().map(|tidx| map[&tidx]).collect_vec(),
            vec!["LP", "INT", "PLUS", "INT", "RP", "TIMES", "INT"]
        );
        assert_eq!(
            spans,
            vec![(0, 1), (1, 1), (3, 1), (5, 2), (7, 1), (8, 1), (10, 1)]
        );
        let (pdfas, map) = get_ab_pdfas();
        let (tokens, spans) = lexer("aabb", &pdfas).unwrap();
        assert_eq!(
            tokens.into_iter().map(|tidx| map[&tidx]).collect_vec(),
            vec!["AA", "BB"]
        );
        assert_eq!(spans, vec![(0, 2), (2, 2)]);
        let (tokens, spans) = lexer("abb", &pdfas).unwrap();
        assert_eq!(
            tokens.into_iter().map(|tidx| map[&tidx]).collect_vec(),
            vec!["AB1", "B"]
        );
        assert_eq!(spans, vec![(0, 2), (2, 1)]);
        assert!(lexer("abac", &pdfas).is_none());
    }

    #[test]
    fn test_prefix_lexer() {
        let (pdfas, map) = get_calc_pfdas();
        let (lexemes, spans, matching, last_span) = prefix_lexer(b"(1 + 28)*\n3", &pdfas).unwrap();
        assert_eq!(
            lexemes.into_iter().map(|tidx| map[&tidx]).collect_vec(),
            vec!["LP", "INT", "PLUS", "INT", "RP", "TIMES"]
        );
        assert_eq!(spans, vec![(0, 1), (1, 1), (3, 1), (5, 2), (7, 1), (8, 1)]);
        assert_eq!(matching.len(), 1);
        assert_eq!(last_span, (10, 1));
        let (idx, state) = matching[0];
        assert_eq!(idx, 4);
        let (pdfa, tidx) = &pdfas[idx];
        assert_eq!(map[tidx.as_ref().unwrap()], "INT");
        assert!(pdfa.is_match_state(state));

        let (lexemes, spans, matching, last_span) = prefix_lexer(b"", &pdfas).unwrap();
        assert!(lexemes.is_empty());
        assert!(spans.is_empty());
        assert_eq!(
            matching,
            pdfas
                .iter()
                .enumerate()
                .map(|(i, (pdfa, _))| (i, pdfa.get_start_state()))
                .collect_vec()
        );
        assert_eq!(last_span, (0, 0));

        let (lexemes, spans, matching, last_span) = prefix_lexer(b"    (", &pdfas).unwrap();
        assert!(lexemes.is_empty());
        assert!(spans.is_empty());
        assert_eq!(matching.len(), 1);
        let (idx, state) = matching[0];
        assert_eq!(idx, 0);
        let (pdfa, tidx) = &pdfas[idx];
        assert_eq!(map[tidx.as_ref().unwrap()], "LP");
        assert!(pdfa.is_match_state(state));
        assert_eq!(last_span, (4, 1));

        let (pdfas, map) = get_ab_pdfas();
        let (lexemes, spans, matching, last_span) = prefix_lexer(b"aabb", &pdfas).unwrap();
        assert_eq!(
            lexemes.into_iter().map(|tidx| map[&tidx]).collect_vec(),
            vec!["AA"]
        );
        assert_eq!(spans, vec![(0, 2)]);
        assert_eq!(matching.len(), 1);
        let (idx, state) = matching[0];
        assert_eq!(idx, 4);
        let (pdfa, tidx) = &pdfas[idx];
        assert_eq!(map[tidx.as_ref().unwrap()], "BB");
        assert!(pdfa.is_match_state(state));
        assert_eq!(last_span, (2, 2));

        let (lexemes, spans, matching, last_span) = prefix_lexer(b"aab", &pdfas).unwrap();
        assert_eq!(
            lexemes.into_iter().map(|tidx| map[&tidx]).collect_vec(),
            vec!["AA"]
        );
        assert_eq!(spans, vec![(0, 2)]);
        assert_eq!(matching.len(), 2);
        let (idx, state) = matching[0];
        assert_eq!(idx, 3);
        let (pdfa, tidx) = &pdfas[idx];
        assert_eq!(map[tidx.as_ref().unwrap()], "B");
        assert!(pdfa.is_match_state(state));
        assert_eq!(last_span, (2, 1));
        let (idx, state) = matching[1];
        assert_eq!(idx, 4);
        let (pdfa, tidx) = &pdfas[idx];
        assert_eq!(map[tidx.as_ref().unwrap()], "BB");
        assert!(!pdfa.is_match_state(state));
    }

    fn load_lrk_grammars() -> Vec<(PathBuf, PathBuf, Vec<PathBuf>)> {
        // list all directories in grammars/
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("grammars");

        let mut grammars = vec![];
        for sub_dir in fs::read_dir(dir).unwrap() {
            let sub_dir = sub_dir.unwrap();
            let sub_dir_name = sub_dir.file_name().to_str().unwrap().to_string();
            // read grammar from .y file in dir/sub_dir/<sub_dir>.y
            let grammar = sub_dir.path().join(format!("{sub_dir_name}.y"));
            // read tokens from .l file in dir/sub_dir/<sub_dir>.l
            let tokens = sub_dir.path().join(format!("{sub_dir_name}.l"));

            // load all examples from dir/sub_dir/examples/
            let examples = fs::read_dir(sub_dir.path().join("examples"))
                .unwrap()
                .map(|entry| entry.unwrap().path())
                .collect();
            grammars.push((grammar, tokens, examples));
        }
        grammars
    }

    #[test]
    fn test_lrk_parser() {
        let grammar = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("grammars/calc/calc.y");
        let tokens = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("grammars/calc/calc.l");
        let lrk = LR1GrammarParser::from_file(grammar, tokens).unwrap();
        assert_eq!(lrk.parse("2 - 1", false), None);
        let text = "(1 + 28)*\n3";
        let parse = lrk.parse(text, false).unwrap();
        println!("{}", parse.pretty(text, true));

        // for (grammar, tokens, examples) in load_lrk_grammars() {
        //     let lrk = LR1GrammarParser::from_file(grammar, tokens).unwrap();
        //     for example in examples {
        //         let text = fs::read_to_string(&example).unwrap();
        //         let parse = lrk.parse(&text, false).unwrap();
        //         println!("{}", parse.pretty(&text, true));
        //     }
        // }
    }

    #[test]
    fn test_lrk_constraint() {
        let conts = load_continuations();

        let grammar = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("grammars/calc/calc.y");
        let tokens = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("grammars/calc/calc.l");
        let lrk = LR1GrammarConstraint::from_file(grammar, tokens, conts.clone()).unwrap();
        let state = lrk.get_start_state();
        let (cont_indices, _) = lrk.get_valid_continuations_with_state(&state);
        println!(
            "matching {}, {} conts: {:#?}",
            lrk.is_match_state(&state),
            cont_indices.len(),
            cont_indices
                .iter()
                .map(|i| String::from_utf8_lossy(&conts[*i]))
                .collect_vec()
        );
        let state = lrk.get_state(b"1").unwrap();
        let (cont_indices, _) = lrk.get_valid_continuations_with_state(&state);
        println!(
            "matching {}, {} conts: {:#?}",
            lrk.is_match_state(&state),
            cont_indices.len(),
            cont_indices
                .iter()
                .take(10)
                .map(|i| String::from_utf8_lossy(&conts[*i]))
                .collect_vec()
        );
        let grammar = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("grammars/test/test.y");
        let tokens = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("grammars/test/test.l");
        let lrk = LR1GrammarConstraint::from_file(grammar, tokens, conts.clone()).unwrap();
        let state = lrk.get_state(b"  SELECT  TEST").unwrap();
        let (cont_indices, _) = lrk.get_valid_continuations_with_state(&state);
        println!(
            "matching {}, {} conts: {:#?}",
            lrk.is_match_state(&state),
            cont_indices.len(),
            cont_indices
                .iter()
                .map(|i| String::from_utf8_lossy(&conts[*i]))
                .collect_vec()
        );
    }
}
