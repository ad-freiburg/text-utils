%start QueryUnit

%%

QueryUnit
    : Query
    ;

QueryType
    : SelectQuery
    | ConstructQuery
    | DescribeQuery
    | AskQuery
    ;

Query
    : Prologue QueryType ValuesClause
    ;

UpdateUnit
    : Update
    ;

PrologueDecl
    : BaseDecl
    | PrefixDecl
    ;

Prologue
    : %empty 
    | PrologueDecl Prologue
    ;

BaseDecl
    : 'BASE' 'IRIREF'
    ;

PrefixDecl
    : 'PREFIX' 'PNAME_NS' 'IRIREF'
    ;

DatasetClauseOptional
    : DatasetClause
    | %empty
    ;

SelectQuery
    : SelectClause DatasetClauseOptional WhereClause SolutionModifier
    ;

SubSelect
    : SelectClause WhereClause SolutionModifier ValuesClause
    ;

DistinctOrReducedOptional
    : DistinctOptional
    | 'REDUCED'
    ;

SelectVar
    : Var
    | '(' Expression 'AS' Var ')'
    ;

SelectVars
    : SelectVar
    | SelectVar SelectVars
    ;

SelectClause
    : 'SELECT' DistinctOrReducedOptional SelectVars 
    | 'SELECT' DistinctOrReducedOptional '*'
    ;

TriplesTemplateOptional
    : TriplesTemplate
    | %empty
    ;

ConstructQuery
    : 'CONSTRUCT' ConstructTemplate DatasetClauseOptional WhereClause SolutionModifier 
    | 'CONSTRUCT' DatasetClauseOptional 'WHERE' '{' TriplesTemplateOptional '}' SolutionModifier
    ;

WhereClauseOptional
    : WhereClause
    | %empty
    ;

VarsOrIris
    : VarOrIri
    | VarOrIri VarsOrIris
    ;

DescribeQuery
    : 'DESCRIBE' VarsOrIris DatasetClauseOptional WhereClauseOptional SolutionModifier
    | 'DESCRIBE' '*' DatasetClauseOptional WhereClauseOptional SolutionModifier
    ;

AskQuery
    : 'ASK' DatasetClauseOptional WhereClause SolutionModifier
    ;

DatasetClause
    : 'FROM' DefaultGraphClause 
    | 'FROM' NamedGraphClause
    ;

DefaultGraphClause
    : SourceSelector
    ;

NamedGraphClause
    : 'NAMED' SourceSelector
    ;

SourceSelector
    : iri
    ;

WhereClause
    : 'WHERE' GroupGraphPattern
    | GroupGraphPattern
    ;

GroupClauseOptional
    : GroupClause
    | %empty
    ;

HavingClauseOptional
    : HavingClause
    | %empty
    ;

OrderClauseOptional
    : OrderClause
    | %empty
    ;

LimitOffsetClausesOptional
    : LimitOffsetClauses
    | %empty
    ;

SolutionModifier
    : GroupClauseOptional HavingClauseOptional OrderClauseOptional LimitOffsetClausesOptional
    ;

GroupConditions
    : GroupCondition
    | GroupCondition GroupConditions
    ;

GroupClause
    : 'GROUP' 'BY' GroupConditions
    ;

GroupCondition
    : BuiltInCall 
    | FunctionCall 
    | '(' Expression ')' 
    | '(' Expression 'AS' Var ')'
    | Var
    ;

HavingConditions
    : HavingCondition
    | HavingCondition HavingConditions
    ;

HavingClause
    : 'HAVING' HavingConditions
    ;

HavingCondition
    : Constraint
    ;

OrderConditions
    : OrderCondition
    | OrderCondition OrderConditions
    ;

OrderClause
    : 'ORDER' 'BY' OrderConditions
    ;

OrderCondition
    : 'ASC' BrackettedExpression
    | 'DESC' BrackettedExpression
    | Constraint
    | Var
    ;

LimitOffsetClauses
    : LimitClause
    | OffsetClause
    | LimitClause OffsetClause
    | OffsetClause LimitClause
    ;

LimitClause
    : 'LIMIT' 'INTEGER'
    ;

OffsetClause
    : 'OFFSET' 'INTEGER'
    ;

ValuesClause
    : %empty
    | 'VALUES' DataBlock
    ;

Update
    : Prologue
    | Prologue Update1
    | Prologue Update1 ';' Update
    ;

Update1
    : Load 
    | Clear 
    | Drop 
    | Add 
    | Move 
    | Copy 
    | Create 
    | InsertData 
    | DeleteData 
    | DeleteWhere 
    | Modify
    ;

SilentOptional
    : 'SILENT'
    | %empty
    ;

Load
    : 'LOAD' SilentOptional iri
    | 'LOAD' SilentOptional iri 'INTO' GraphRef
    ;

Clear
    : 'CLEAR' SilentOptional GraphRefAll
    ;

Drop
    : 'DROP' SilentOptional GraphRefAll
    ;

Create
    : 'CREATE' SilentOptional GraphRef
    ;

Add
    : 'ADD' SilentOptional GraphOrDefault 'TO' GraphOrDefault
    ;

Move
    : 'MOVE' SilentOptional GraphOrDefault 'TO' GraphOrDefault
    ;

Copy
    : 'COPY' SilentOptional GraphOrDefault 'TO' GraphOrDefault
    ;

InsertData
    : 'INSERT' 'DATA' QuadData
    ;

DeleteData
    : 'DELETE' 'DATA' QuadData
    ;

DeleteWhere
    : 'DELETE' 'WHERE' QuadPattern
    ;

UsingClausesOptional
    : %empty
    | UsingClause UsingClausesOptional
    ;

DeleteOrInsertClauses
    : DeleteClause
    | InsertClause
    | DeleteClause InsertClause
    ;

Modify
    : 'WITH' iri DeleteOrInsertClauses UsingClausesOptional 'WHERE' GroupGraphPattern
    | DeleteOrInsertClauses UsingClausesOptional 'WHERE' GroupGraphPattern
    ;

DeleteClause
    : 'DELETE' QuadPattern
    ;

InsertClause
    : 'INSERT' QuadPattern
    ;

UsingClause
    : 'USING' iri 
    | 'USING' 'NAMED' iri
    ;

GraphOrDefault
    : 'DEFAULT' 
    | iri
    | 'GRAPH' iri
    ;

GraphRef
    : 'GRAPH' iri
    ;

GraphRefAll
    : GraphRef 
    | 'DEFAULT' 
    | 'NAMED' 
    | 'ALL'
    ;

QuadPattern
    : '{' Quads '}'
    ;

QuadData
    : '{' Quads '}'
    ;

DotOptional
    : '.'
    | %empty
    ;

QuadsOptional
    : %empty
    | QuadsNotTriples DotOptional TriplesTemplateOptional QuadsOptional
    ;

Quads
    : TriplesTemplateOptional QuadsOptional
    ;

QuadsNotTriples
    : 'GRAPH' VarOrIri '{' TriplesTemplateOptional '}'
    ;

TriplesTemplate
    : TriplesSameSubject
    | TriplesSameSubject '.' TriplesTemplateOptional
    ;

GroupGraphPattern
    : '{' SubSelect '}'
    | '{' GroupGraphPatternSub '}'
    ;

TriplesBlockOptional
    : %empty
    | TriplesBlock
    ;

GroupGraphPatternSubOptional
    : %empty
    | GraphPatternNotTriples DotOptional TriplesBlockOptional GroupGraphPatternSubOptional
    ;

GroupGraphPatternSub
    : TriplesBlockOptional GroupGraphPatternSubOptional
    ;

TriplesBlock
    : TriplesSameSubjectPath
    | TriplesSameSubjectPath '.' TriplesBlockOptional
    ;

GraphPatternNotTriples
    : GroupOrUnionGraphPattern 
    | OptionalGraphPattern 
    | MinusGraphPattern 
    | GraphGraphPattern 
    | ServiceGraphPattern 
    | Filter 
    | Bind 
    | InlineData
    ;

OptionalGraphPattern
    : 'OPTIONAL' GroupGraphPattern
    ;

GraphGraphPattern
    : 'GRAPH' VarOrIri GroupGraphPattern
    ;

ServiceGraphPattern
    : 'SERVICE' SilentOptional VarOrIri GroupGraphPattern
    ;

Bind
    : 'BIND' '(' Expression 'AS' Var ')'
    ;

InlineData
    : 'VALUES' DataBlock
    ;

DataBlock
    : InlineDataOneVar 
    | InlineDataFull
    ;

DataBlockValuesOptional
    : %empty
    | DataBlockValue DataBlockValuesOptional
    ;

InlineDataOneVar
    : Var '{' DataBlockValuesOptional '}'
    ;

VarsOptional
    : %empty
    | Var VarsOptional
    ;

NilOrDataBlockValuesOptional
    : %empty
    | 'NIL' NilOrDataBlockValuesOptional
    | '(' DataBlockValuesOptional ')' NilOrDataBlockValuesOptional
    ;

InlineDataFull
    : 'NIL' '{' NilOrDataBlockValuesOptional '}'
    | '(' VarsOptional ')' '{' NilOrDataBlockValuesOptional '}'
    ;

DataBlockValue
    : iri 
    | RDFLiteral 
    | NumericLiteral 
    | BooleanLiteral 
    | 'UNDEF'
    ;

MinusGraphPattern
    : 'MINUS' GroupGraphPattern
    ;

GroupOrUnionGraphPattern
    : GroupGraphPattern 
    | GroupGraphPattern 'UNION' GroupOrUnionGraphPattern
    ;

Filter
    : 'FILTER' Constraint
    ;

Constraint
    : BrackettedExpression 
    | BuiltInCall 
    | FunctionCall
    ;

FunctionCall
    : iri ArgList
    ;

Expressions
    : Expression
    | Expression ',' Expressions
    ;

ArgList
    : 'NIL' 
    | '(' 'DISTINCT' Expressions ')'
    | '(' Expressions ')'
    ;

ExpressionList
    : 'NIL' 
    | '(' Expressions ')'
    ;

ConstructTriplesOptional
    : %empty
    | ConstructTriples
    ;

ConstructTemplate
    : '{' ConstructTriplesOptional '}'
    ;

ConstructTriples
    : TriplesSameSubject
    | TriplesSameSubject '.' ConstructTriplesOptional
    ;

TriplesSameSubject
    : VarOrTerm PropertyListNotEmpty 
    | TriplesNode PropertyList
    ;

PropertyList
    : %empty
    | PropertyListNotEmpty
    ;

PropertyListNotEmpty
    : Verb ObjectList
    | Verb ObjectList ';'
    | Verb ObjectList ';' PropertyListNotEmpty
    ;

Verb
    : VarOrIri 
    | 'a'
    ;

ObjectList
    : Object
    | Object ',' ObjectList
    ;

Object
    : GraphNode
    ;

TriplesSameSubjectPath
    : VarOrTerm PropertyListPathNotEmpty 
    | TriplesNodePath PropertyListPath
    ;

PropertyListPath
    : %empty
    | PropertyListPathNotEmpty
    ;

VerbPathOrSimple
    : VerbPath 
    | VerbSimple
    ;

VerbPathObjectListOptional
    : %empty
    | ';' VerbPathObjectListOptional
    | ';' VerbPathOrSimple ObjectList VerbPathObjectListOptional
    ;

PropertyListPathNotEmpty
    : VerbPathOrSimple ObjectListPath VerbPathObjectListOptional
    ;

VerbPath
    : Path
    ;

VerbSimple
    : Var
    ;

ObjectListPath
    : ObjectPath
    | ObjectPath ',' ObjectListPath
    ;

ObjectPath
    : GraphNodePath
    ;

Path
    : PathAlternative
    ;

PathAlternative
    : PathSequence
    | PathSequence '|' PathAlternative
    ;

PathSequence
    : PathEltOrInverse
    | PathEltOrInverse '/' PathSequence
    ;

PathElt
    : PathPrimary
    | PathPrimary PathMod
    ;

PathEltOrInverse
    : PathElt 
    | '^' PathElt
    ;

PathMod
    : '?' 
    | '*' 
    | '+'
    ;

PathPrimary
    : iri 
    | 'a' 
    | '!' PathNegatedPropertySet 
    | '(' Path ')'
    ;

PathOneInPropertySets
    : PathOneInPropertySet
    | PathOneInPropertySet '|' PathOneInPropertySets
    ;

PathNegatedPropertySet
    : PathOneInPropertySet 
    | '(' ')'
    | '(' PathOneInPropertySets ')'
    ;

PathOneInPropertySet
    : iri 
    | 'a' 
    | '^' iri
    | '^' 'a'
    ;

TriplesNode
    : Collection 
    | BlankNodePropertyList
    ;

BlankNodePropertyList
    : '[' PropertyListNotEmpty ']'
    ;

TriplesNodePath
    : CollectionPath 
    | BlankNodePropertyListPath
    ;

BlankNodePropertyListPath
    : '[' PropertyListPathNotEmpty ']'
    ;

GraphNodes
    : GraphNode
    | GraphNode GraphNodes
    ;

Collection
    : '(' GraphNodes ')'
    ;

GraphNodePaths
    : GraphNodePath
    | GraphNodePath GraphNodePaths
    ;

CollectionPath
    : '(' GraphNodePaths ')'
    ;

GraphNode
    : VarOrTerm 
    | TriplesNode
    ;

GraphNodePath
    : VarOrTerm 
    | TriplesNodePath
    ;

VarOrTerm
    : Var 
    | GraphTerm
    ;

VarOrIri
    : Var 
    | iri
    ;

Var
    : 'VAR1' 
    | 'VAR2'
    ;

GraphTerm
    : iri 
    | RDFLiteral 
    | NumericLiteral 
    | BooleanLiteral 
    | BlankNode 
    | 'NIL'
    ;

Expression
    : ConditionalOrExpression
    ;

ConditionalOrExpression
    : ConditionalAndExpression
    | ConditionalAndExpression '||' ConditionalOrExpression
    ;

ConditionalAndExpression
    : ValueLogical
    | ValueLogical '&&' ConditionalAndExpression
    ;

ValueLogical
    : RelationalExpression
    ;

ComparisonOp
    : '=' 
    | '!=' 
    | '<' 
    | '>' 
    | '<=' 
    | '>=' 
    ;

ContainmentOp
    : 'IN' 
    | 'NOT' 'IN'
    ;

RelationalExpression
    : NumericExpression
    | NumericExpression ComparisonOp NumericExpression 
    | NumericExpression ContainmentOp ExpressionList
    ;

NumericExpression
    : AdditiveExpression
    ;

MulOrDiv
    : '*' UnaryExpression
    | '/' UnaryExpression
    ;

MulsOrDivsOptional
    : %empty
    | MulOrDiv MulsOrDivsOptional
    ;

RhsAdditiveExpressionsOptional
    : %empty
    | '+' MultiplicativeExpression RhsAdditiveExpressionsOptional
    | '-' MultiplicativeExpression RhsAdditiveExpressionsOptional
    | NumericLiteralPositive MulsOrDivsOptional RhsAdditiveExpressionsOptional
    | NumericLiteralNegative MulsOrDivsOptional RhsAdditiveExpressionsOptional
    ;

AdditiveExpression
    : MultiplicativeExpression RhsAdditiveExpressionsOptional
    ;

MultiplicativeExpression
    : UnaryExpression MulsOrDivsOptional
    ;

UnaryExpression
    : '!' PrimaryExpression
    | '+' PrimaryExpression
    | '-' PrimaryExpression
    | PrimaryExpression
    ;

PrimaryExpression
    : BrackettedExpression 
    | BuiltInCall 
    | iriOrFunction 
    | RDFLiteral 
    | NumericLiteral 
    | BooleanLiteral 
    | Var
    ;

BrackettedExpression
    : '(' Expression ')'
    ;

BuiltInCall
    : Aggregate
    | 'STR' '(' Expression ')'
    | 'LANG' '(' Expression ')'
    | 'LANGMATCHES' '(' Expression ',' Expression ')'
    | 'DATATYPE' '(' Expression ')'
    | 'BOUND' '(' Var ')'
    | 'IRI' '(' Expression ')'
    | 'URI' '(' Expression ')'
    | 'BNODE' 'NIL'
    | 'BNODE' '(' Expression ')'
    | 'RAND' 'NIL'
    | 'ABS' '(' Expression ')'
    | 'CEIL' '(' Expression ')'
    | 'FLOOR' '(' Expression ')'
    | 'ROUND' '(' Expression ')'
    | 'CONCAT' ExpressionList
    | SubstringExpression
    | 'STRLEN' '(' Expression ')'
    | StrReplaceExpression
    | 'UCASE' '(' Expression ')'
    | 'LCASE' '(' Expression ')'
    | 'ENCODE_FOR_URI' '(' Expression ')'
    | 'CONTAINS' '(' Expression ',' Expression ')'
    | 'STRSTARTS' '(' Expression ',' Expression ')'
    | 'STRENDS' '(' Expression ',' Expression ')'
    | 'STRBEFORE' '(' Expression ',' Expression ')'
    | 'STRAFTER' '(' Expression ',' Expression ')'
    | 'YEAR' '(' Expression ')'
    | 'MONTH' '(' Expression ')'
    | 'DAY' '(' Expression ')'
    | 'HOURS' '(' Expression ')'
    | 'MINUTES' '(' Expression ')'
    | 'SECONDS' '(' Expression ')'
    | 'TIMEZONE' '(' Expression ')'
    | 'TZ' '(' Expression ')'
    | 'NOW' 'NIL'
    | 'UUID' 'NIL'
    | 'STRUUID' 'NIL'
    | 'MD5' '(' Expression ')'
    | 'SHA1' '(' Expression ')'
    | 'SHA256' '(' Expression ')'
    | 'SHA384' '(' Expression ')'
    | 'SHA512' '(' Expression ')'
    | 'COALESCE' ExpressionList
    | 'IF' '(' Expression ',' Expression ',' Expression ')'
    | 'STRLANG' '(' Expression ',' Expression ')'
    | 'STRDT' '(' Expression ',' Expression ')'
    | 'SAMETERM' '(' Expression ',' Expression ')'
    | 'ISIRI' '(' Expression ')'
    | 'ISURI' '(' Expression ')'
    | 'ISBLANK' '(' Expression ')'
    | 'ISLITERAL' '(' Expression ')'
    | 'ISNUMERIC' '(' Expression ')'
    | RegexExpression
    | ExistsFunc
    | NotExistsFunc
    ;

RegexExpression
    : 'REGEX' '(' Expression ',' Expression ')'
    | 'REGEX' '(' Expression ',' Expression ',' Expression ')'
    ;

SubstringExpression
    : 'SUBSTR' '(' Expression ',' Expression ')'
    | 'SUBSTR' '(' Expression ',' Expression ',' Expression ')'
    ;

StrReplaceExpression
    : 'REPLACE' '(' Expression ',' Expression ',' Expression ')'
    | 'REPLACE' '(' Expression ',' Expression ',' Expression ',' Expression ')'
    ;
    
ExistsFunc
    : 'EXISTS' GroupGraphPattern
    ;

NotExistsFunc
    : 'NOT' 'EXISTS' GroupGraphPattern
    ;

DistinctOptional
    : 'DISTINCT'
    | %empty
    ;

Aggregate
    : 'COUNT' '(' DistinctOptional Expression ')'
    | 'COUNT' '(' DistinctOptional '*' ')'
    | 'SUM' '(' DistinctOptional Expression ')'
    | 'MIN' '(' DistinctOptional Expression ')'
    | 'MAX' '(' DistinctOptional Expression ')'
    | 'AVG' '(' DistinctOptional Expression ')'
    | 'SAMPLE' '(' DistinctOptional Expression ')'
    | 'GROUP_CONCAT' '(' DistinctOptional Expression ')'
    | 'GROUP_CONCAT' '(' DistinctOptional Expression ';' 'SEPARATOR' '=' String ')'
    ;

iriOrFunction
    : iri 
    | iri ArgList
    ;

RDFLiteral
    : String
    | String 'LANGTAG'
    | String '^^' iri
    ;

NumericLiteral
    : NumericLiteralUnsigned 
    | NumericLiteralPositive 
    | NumericLiteralNegative
    ;

NumericLiteralUnsigned
    : 'INTEGER' 
    | 'DECIMAL' 
    | 'DOUBLE'
    ;

NumericLiteralPositive
    : 'INTEGER_POSITIVE' 
    | 'DECIMAL_POSITIVE' 
    | 'DOUBLE_POSITIVE'
    ;

NumericLiteralNegative
    : 'INTEGER_NEGATIVE' 
    | 'DECIMAL_NEGATIVE' 
    | 'DOUBLE_NEGATIVE'
    ;

BooleanLiteral
    : 'true' 
    | 'false'
    ;

String
    : 'STRING_LITERAL1' 
    | 'STRING_LITERAL2' 
    | 'STRING_LITERAL_LONG1' 
    | 'STRING_LITERAL_LONG2'
    ;

iri
    : 'IRIREF' 
    | PrefixedName
    ;

PrefixedName
    : 'PNAME_LN' 
    | 'PNAME_NS'
    ;

BlankNode
    : 'BLANK_NODE_LABEL' 
    | 'ANON'
    ;
