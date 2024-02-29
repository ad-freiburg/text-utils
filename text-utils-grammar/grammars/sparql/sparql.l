// directly ported from https://www.w3.org/TR/sparql11-query/#sparqlGrammar

// fragments
EXPONENT [eE] [+-]? [0-9]+
ECHAR '\' [tbnrf\\"']
WS \x20 | \x09 | \x0D | \x0A
PN_CHARS_BASE [A-Z] | [a-z] | [\u00C0-\u00D6] | [\u00D8-\u00F6] | [\u00F8-\u02FF] | [\u0370-\u037D] | [\u037F-\u1FFF] | [\u200C-\u200D] | [\u2070-\u218F] | [\u2C00-\u2FEF] | [\u3001-\uD7FF] | [\uF900-\uFDCF] | [\uFDF0-\uFFFD] | [\U00010000-\U000EFFFF]
PN_CHARS_U {PN_CHARS_BASE} | '_'
VARNAME ( {PN_CHARS_U} | [0-9] ) ( {PN_CHARS_U} | [0-9] | \u00B7 | [\u0300-\u036F] | [\u203F-\u2040] )*
PN_CHARS {PN_CHARS_U} | '-' | [0-9] | \u00B7 | [\u0300-\u036F] | [\u203F-\u2040]
PN_PREFIX {PN_CHARS_BASE} (( {PN_CHARS} | '.' )* {PN_CHARS})?
PN_LOCAL ({PN_CHARS_U} | ':' | [0-9] | {PLX} ) (({PN_CHARS} | '.' | ':' | {PLX})* ({PN_CHARS} | ':' | {PLX}) )?
PLX {PERCENT} | {PN_LOCAL_ESC}
PERCENT '%' {HEX} {HEX}
HEX [0-9] | [A-F] | [a-f]
PN_LOCAL_ESC '\' ( '_' | '~' | '.' | '-' | '!' | '$' | '&' | "'" | '(' | ')' | '*' | '+' | ',' | ';' | '=' | '/' | '?' | '#' | '@' | '%' )

%%

// SPARQL keywords
BASE        (?i)BASE
PREFIX      (?i)PREFIX
DISTINCT    (?i)DISTINCT
REDUCED     (?i)REDUCED
SELECT      (?i)SELECT
AS          (?i)AS
CONSTRUCT   (?i)CONSTRUCT
WHERE       (?i)WHERE
DESCRIBE    (?i)DESCRIBE
ASK         (?i)ASK
NAMED       (?i)NAMED
GROUP       (?i)GROUP
BY          (?i)BY
HAVING      (?i)HAVING
ORDER       (?i)ORDER
ASC         (?i)ASC
DESC        (?i)DESC
LIMIT       (?i)LIMIT
OFFSET      (?i)OFFSET
VALUES      (?i)VALUES
LOAD        (?i)LOAD
SILENT      (?i)SILENT
INTO        (?i)INTO
CLEAR       (?i)CLEAR
DROP        (?i)DROP
CREATE      (?i)CREATE
ADD         (?i)ADD
TO          (?i)TO
MOVE        (?i)MOVE
COPY        (?i)COPY
INSERT      (?i)INSERT
DELETE      (?i)DELETE
DATA        (?i)DATA
WITH        (?i)WITH
USING       (?i)USING
DEFAULT     (?i)DEFAULT
GRAPH       (?i)GRAPH
ALL         (?i)ALL
OPTIONAL    (?i)OPTIONAL
SERVICE     (?i)SERVICE
BIND        (?i)BIND
UNDEF       (?i)UNDEF
MINUS       (?i)MINUS
UNION       (?i)UNION
FILTER      (?i)FILTER
IN          (?i)IN
NOT         (?i)NOT
STR         (?i)STR
LANG        (?i)LANG
LANGMATCHES (?i)LANGMATCHES
DATATYPE    (?i)DATATYPE
BOUND       (?i)BOUND
IRI         (?i)IRI
URI         (?i)URI
BNODE       (?i)BNODE
RAND        (?i)RAND
ABS         (?i)ABS
CEIL        (?i)CEIL
FLOOR       (?i)FLOOR
ROUND       (?i)ROUND
CONCAT      (?i)CONCAT
STRLEN      (?i)STRLEN
UCASE       (?i)UCASE
LCASE       (?i)LCASE
ENCODE_FOR_URI (?i)ENCODE_FOR_URI
CONTAINS    (?i)CONTAINS
STRSTARTS   (?i)STRSTARTS
STRENDS     (?i)STRENDS
STRBEFORE   (?i)STRBEFORE
STRAFTER    (?i)STRAFTER
YEAR        (?i)YEAR
MONTH       (?i)MONTH
DAY         (?i)DAY
HOURS       (?i)HOURS
MINUTES     (?i)MINUTES
SECONDS     (?i)SECONDS
TIMEZONE    (?i)TIMEZONE
TZ          (?i)TZ
NOW         (?i)NOW
UUID        (?i)UUID
STRUUID     (?i)STRUUID
MD5         (?i)MD5
SHA1        (?i)SHA1
SHA256      (?i)SHA256
SHA384      (?i)SHA384
SHA512      (?i)SHA512
COALESCE    (?i)COALESCE
IF          (?i)IF
STRLANG     (?i)STRLANG
STRDT       (?i)STRDT
SAMETERM    (?i)sameTerm
ISIRI       (?i)isIRI
ISURI       (?i)isURI
ISBLANK     (?i)isBLANK
ISLITERAL   (?i)isLITERAL
ISNUMERIC   (?i)isNUMERIC
REGEX       (?i)REGEX
SUBSTR      (?i)SUBSTR
REPLACE     (?i)REPLACE
EXISTS      (?i)EXISTS
COUNT       (?i)COUNT
SUM         (?i)SUM
MIN         (?i)MIN
MAX         (?i)MAX
AVG         (?i)AVG
SAMPLE      (?i)SAMPLE
GROUP_CONCAT (?i)GROUP_CONCAT
SEPARATOR   (?i)SEPARATOR

// Misc

IRIREF '<' ([[^<>"{}|^`\\]--[\x00-\x20]])* '>'
PNAME_NS {PN_PREFIX}? ':'
PNAME_LN {PNAME_NS} {PN_LOCAL}
BLANK_NODE_LABEL '_:' ({PN_CHARS_U}|[0-9])(( {PN_CHARS} | '.' )*{PN_CHARS})?
VAR1 '?' {VARNAME}
VAR2 '$' {VARNAME}
LANGTAG '@' [a-zA-Z]+ (-[a-zA-Z0-9]+)*
INTEGER [0-9]+
DECIMAL [0-9]* '.' [0-9]+
DOUBLE [0-9]+ '.' [0-9]* {EXPONENT} | '.' ([0-9])+ {EXPONENT} | ([0-9])+ {EXPONENT}
INTEGER_POSITIVE '+' {INTEGER}
DECIMAL_POSITIVE '+' {DECIMAL}
DOUBLE_POSITIVE '+' {DOUBLE}
INTEGER_NEGATIVE '-' {INTEGER}
DECIMAL_NEGATIVE '-' {DECIMAL}
DOUBLE_NEGATIVE '-' {DOUBLE}
STRING_LITERAL1 "'" ( ([^\x27\x5C\x0A\x0D]) | {ECHAR} )* "'"
STRING_LITERAL2 '"' ( ([^\x22\x5C\x0A\x0D]) | {ECHAR} )* '"'
STRING_LITERAL_LONG1 "'''" ( ( "'" | "''" )? ( [^'\\] | {ECHAR} ) )* "'''"
STRING_LITERAL_LONG2 '"""' ( ( '"' | '""' )? ( [^"\\] | {ECHAR} ) )* '"""'
NIL '(' {WS}* ')'
ANON '[' {WS}* ']'

// ignore whitespace and comments
; {WS}+
; '#' [^\x0D\x0A]*
