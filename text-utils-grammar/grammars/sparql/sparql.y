%start query
%%
query
    : prologue selectOrAsk
    ;

selectOrAsk
    : selectQuery
    | askQuery
    ;

baseDeclOptional
    : %empty
    | baseDecl
    ;

prefixDeclStar
    : %empty
    | prefixDecl prefixDeclStar
    ;

prologue
    : baseDeclOptional prefixDeclStar
    ;

baseDecl
    : 'BASE' 'IRI_REF'
    ;

prefixDecl
    : 'PREFIX' 'PNAME_NS' 'IRI_REF'
    ;

distinctOrReducedOptional
    : %empty
    | 'DISTINCT'
    | 'REDUCED'
    ;

datasetClauseStar
    : %empty
    | datasetClause datasetClauseStar
    ;

var_Plus
    : var_
    | var_ var_Plus
    ;

var_PlusOrStar
    : var_Plus
    | '*'
    ;

selectQuery
    : 'SELECT' distinctOrReducedOptional var_PlusOrStar datasetClauseStar whereClause solutionModifier
    ;

askQuery
    : 'ASK' datasetClauseStar whereClause
    ;

defaultGraphClauseOrNamedGraphClause
    : defaultGraphClause
    | namedGraphClause
    ;

datasetClause
    : 'FROM' defaultGraphClauseOrNamedGraphClause
    ;

defaultGraphClause
    : sourceSelector
    ;

namedGraphClause
    : 'NAMED' sourceSelector
    ;

sourceSelector
    : iriRef
    ;

whereClause
    : groupGraphPattern
    | 'WHERE' groupGraphPattern
    ;

solutionModifier
    : %empty
    | orderClause
    | limitOffsetClauses
    | orderClause limitOffsetClauses
    ;

limitOffsetClauses
    : limitClause
    | offsetClause
    | limitClause offsetClause
    | offsetClause limitClause
    ;

orderConditionPlus
    : orderCondition
    | orderCondition orderConditionPlus
    ;

orderClause
    : 'ORDER' 'BY' orderConditionPlus
    ;

ascOrDesc
    : 'ASC'
    | 'DESC'
    ;

orderCondition
    : ascOrDesc brackettedExpression
    | constraint
    | var_
    ;

limitClause
    : 'LIMIT' 'INTEGER'
    ;

offsetClause
    : 'OFFSET' 'INTEGER'
    ;

triplesBlockOptional
    : %empty
    | triplesBlock
    ;

graphPatternNotTriplesOrFilter
    : graphPatternNotTriples
    | filter_
    ;

dotOptional
    : %empty
    | '.'
    ;

graphFilterDotTriplesStar
    : %empty
    | graphPatternNotTriplesOrFilter dotOptional triplesBlockOptional graphFilterDotTriplesStar
    ;

groupGraphPattern
    : '{' triplesBlockOptional graphFilterDotTriplesStar '}'
    ;

triplesBlock
    : triplesSameSubject
    | triplesSameSubject '.' triplesBlockOptional
    ;

graphPatternNotTriples
    : optionalGraphPattern
    | groupOrUnionGraphPattern
    | graphGraphPattern
    ;

optionalGraphPattern
    : 'OPTIONAL' groupGraphPattern
    ;

graphGraphPattern
    : 'GRAPH' varOrIRIref groupGraphPattern
    ;

unionGroupGraphPatternStar
    : %empty
    | 'UNION' groupGraphPattern unionGroupGraphPatternStar
    ;

groupOrUnionGraphPattern
    : groupGraphPattern unionGroupGraphPatternStar
    ;

filter_
    : 'FILTER' constraint
    ;

constraint
    : brackettedExpression
    | builtInCall
    | functionCall
    ;

functionCall
    : iriRef argList
    ;

commaExpressionStar
    : %empty
    | ',' expression commaExpressionStar
    ;

argList
    : 'NIL'
    | '(' expression commaExpressionStar ')'
    ;

triplesSameSubject
    : varOrTerm propertyListNotEmpty
    | triplesNode propertyList
    ;

verbObjectListOptional
    : %empty
    | verb objectList
    ;

semicolonVerbObjectListOptionalStar
    : %empty
    | ';' verbObjectListOptional semicolonVerbObjectListOptionalStar
    ;

propertyListNotEmpty
    : verb objectList semicolonVerbObjectListOptionalStar
    ;

propertyList
    : %empty
    | propertyListNotEmpty
    ;

objectList
    : object_
    | object_ ',' objectList
    ;

object_
    : graphNode
    ;

verb
    : varOrIRIref
    | 'A'
    ;

triplesNode
    : collection
    | blankNodePropertyList
    ;

blankNodePropertyList
    : '[' propertyListNotEmpty ']'
    ;

graphNodePlus
    : graphNode
    | graphNode graphNodePlus
    ;

collection
    : '(' graphNodePlus ')'
    ;

graphNode
    : varOrTerm
    | triplesNode
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

expression
    : conditionalOrExpression
    ;

conditionalOrExpression
    : conditionalAndExpression
    | conditionalAndExpression '||' conditionalOrExpression
    ;

conditionalAndExpression
    : valueLogical
    | valueLogical '&&' conditionalAndExpression
    ;

valueLogical
    : relationalExpression
    ;

relationalOp
    : '='
    | '!='
    | '<'
    | '>'
    | '<='
    | '>='
    ;

relationalExpression
    : numericExpression
    | numericExpression relationalOp numericExpression
    ;

numericExpression
    : additiveExpression
    ;

plusOrMinus
    : '+'
    | '-'
    ;

additiveExpression
    : multiplicativeExpression 
    | plusOrMinus multiplicativeExpression additiveExpression
    | numericLiteralPositive additiveExpression
    | numericLiteralNegative additiveExpression
    ;

mulOrDiv
    : '*'
    | '/'
    ;

multiplicativeExpression
    : unaryExpression 
    | unaryExpression mulOrDiv multiplicativeExpression
    ;

unaryExpression
    : '!' primaryExpression
    | '+' primaryExpression
    | '-' primaryExpression
    | primaryExpression
    ;

primaryExpression
    : brackettedExpression
    | builtInCall
    | iriRefOrFunction
    | rdfLiteral
    | numericLiteral
    | booleanLiteral
    | var_
    ;

brackettedExpression
    : '(' expression ')'
    ;

builtInCall
    : 'STR' '(' expression ')'
    | 'LANG' '(' expression ')'
    | 'LANGMATCHES' '(' expression ',' expression ')'
    | 'DATATYPE' '(' expression ')'
    | 'BOUND' '(' var_ ')'
    | 'SAME_TERM' '(' expression ',' expression ')'
    | 'IS_IRI' '(' expression ')'
    | 'IS_URI' '(' expression ')'
    | 'IS_BLANK' '(' expression ')'
    | 'IS_LITERAL' '(' expression ')'
    | regexExpression
    ;

regexExpression
    : 'REGEX' '(' expression ',' expression ')'
    | 'REGEX' '(' expression ',' expression ',' expression ')'
    ;

iriRefOrFunction
    : iriRef 
    | iriRef argList
    ;

rdfLiteral
    : string_
    | string_ 'LANGTAG'
    | string_ '^^' iriRef
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
