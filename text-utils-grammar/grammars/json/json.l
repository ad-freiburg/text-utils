// ported from antlr4 grammar at https://github.com/antlr/grammars-v4/tree/master/json
HEX [0-9a-fA-F]
UNICODE u{HEX}{4}
ESC '\' ({UNICODE}|["\\/bfnrt])
SAFECODEPOINT [^\x00-\x1F"\\]
INT 0|([1-9][0-9]*)
EXP [Ee][+-]?[0-9]+
// need to write \x20 instead of literal space because pattern parts
// are split by whitespaces
WS [\x20\t\n\r]+

%%

STRING '"' ({ESC}|{SAFECODEPOINT})* '"'
NUMBER -?{INT}( '.' [0-9]+)?{EXP}?
; {WS}
