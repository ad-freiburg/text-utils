// fragments
WS [\x20\t\n\r]+
DIGIT [0-9]
PN_CHARS_BASE [A-Za-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD]
PN_CHARS_U {PN_CHARS_BASE}|_
PN_CHARS {PN_CHARS_U}|-|{DIGIT}
VARNAME ({PN_CHARS_U}|{DIGIT})({PN_CHARS_U}|{DIGIT}|\u00B7|[\u0300-\u036F]|[\u203F-\u2040])*
EXPONENT 'E' [+-]? {DIGIT}+
ECHAR '\' (?i)[tbnrf"']
PN_PREFIX {PN_CHARS_BASE}(({PN_CHARS}|\.)*{PN_CHARS})?
PN_LOCAL ({PN_CHARS_U}|{DIGIT})(({PN_CHARS}|\.)*{PN_CHARS})?

%%

A           'a'
// all keywords except a case insensitive
ASC         (?i)ASC
ASK         (?i)ASK
BASE        (?i)BASE
BOUND       (?i)BOUND
BY          (?i)BY
// CONSTRUCT   (?i)CONSTRUCT
DATATYPE    (?i)DATATYPE
DESC        (?i)DESC
// DESCRIBE    (?i)DESCRIBE
DISTINCT    (?i)DISTINCT
FILTER      (?i)FILTER
FROM        (?i)FROM
GRAPH       (?i)GRAPH
LANG        (?i)LANG
LANGMATCHES (?i)LANGMATCHES
LIMIT       (?i)LIMIT
NAMED       (?i)NAMED
OFFSET      (?i)OFFSET
OPTIONAL    (?i)OPTIONAL
ORDER       (?i)ORDER
PREFIX      (?i)PREFIX
REDUCED     (?i)REDUCED
REGEX       (?i)REGEX
SELECT      (?i)SELECT
STR         (?i)STR
UNION       (?i)UNION
WHERE       (?i)WHERE
TRUE        (?i)true
FALSE       (?i)false

IS_LITERAL  (?i)isLITERAL
IS_BLANK    (?i)isBLANK
IS_URI      (?i)isURI
IS_IRI      (?i)isIRI
SAME_TERM   (?i)sameTerm

// MISC

IRI_REF '<' ([^<>"\{\}|^\\`]|PN_CHARS)* '>'

PNAME_NS {PN_PREFIX}? ':'

PNAME_LN {PNAME_NS} {PN_LOCAL}

BLANK_NODE_LABEL '_:' {PN_LOCAL}

VAR1 '?' {VARNAME}

VAR2 '$' {VARNAME}

LANGTAG '@' {PN_CHARS_BASE}+ (-({PN_CHARS_BASE}|{DIGIT})+)*

INTEGER {DIGIT}+

DECIMAL {DIGIT}+\.{DIGIT}*|\.{DIGIT}+

DOUBLE {DIGIT}+\.{DIGIT}*{EXPONENT}|\.?{DIGIT}+{EXPONENT}

INTEGER_POSITIVE '+' {INTEGER}

DECIMAL_POSITIVE '+' {DECIMAL}

DOUBLE_POSITIVE '+' {DOUBLE}

INTEGER_NEGATIVE '-' {INTEGER}

DECIMAL_NEGATIVE '-' {DECIMAL}

DOUBLE_NEGATIVE '-' {DOUBLE}

STRING_LITERAL1 "'" ([^\u0027\u005C\u000A\u000D]|{ECHAR})* "'"

STRING_LITERAL2 '"' ([^\u0027\u005C\u000A\u000D]|{ECHAR})* '"'

// STRING_LITERAL_LONG1 "'''" (('|'')?(^['\\]|{ECHAR}))* "'''"

// STRING_LITERAL_LONG2 '"""' (("|"")?(^["\\]|{ECHAR}))* '"""'

NIL '(' {WS}* ')'

ANON '[' {WS}* ']'

; {WS}
