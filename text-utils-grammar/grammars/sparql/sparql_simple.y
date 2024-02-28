%start query
%%
query
    : selectQuery
    ;

selectQuery
    : 'SELECT' '*' whereClause
    ;

whereClause
    : 'WHERE' groupGraphPattern
    ;

groupGraphPattern
    : '{' triplesBlock_plus '}'
    ;

triplesBlock_plus
    : triplesBlock
    | triplesBlock '.'
    | triplesBlock '.' triplesBlock_plus
    ;

triplesBlock
    : varOrTerm varOrIRIref varOrIRIref
    ;

varOrTerm
    : var_
    | graphTerm
    ;

varOrIRIref
    : var_
    | iriRef
    ;

var_
    : 'VAR1'
    | 'VAR2'
    ;

graphTerm
    : iriRef
    | rdfLiteral
    | numericLiteral
    | booleanLiteral
    | blankNode
    | 'NIL'
    ;

rdfLiteral
    : string_
    ;

numericLiteral
    : numericLiteralUnsigned
    | numericLiteralPositive
    | numericLiteralNegative
    ;

numericLiteralUnsigned
    : 'INTEGER'
    | 'DECIMAL'
    | 'DOUBLE'
    ;

numericLiteralPositive
    : 'INTEGER_POSITIVE'
    | 'DECIMAL_POSITIVE'
    | 'DOUBLE_POSITIVE'
    ;

numericLiteralNegative
    : 'INTEGER_NEGATIVE'
    | 'DECIMAL_NEGATIVE'
    | 'DOUBLE_NEGATIVE'
    ;

booleanLiteral
    : 'TRUE'
    | 'FALSE'
    ;

string_
    : 'STRING_LITERAL1'
    | 'STRING_LITERAL2'
    ;

iriRef
    : 'IRI_REF'
    | prefixedName
    ;

prefixedName
    : 'PNAME_LN'
    | 'PNAME_NS'
    ;

blankNode
    : 'BLANK_NODE_LABEL'
    | 'ANON'
    ;
