%start Expr

%%

Expr: Expr '+' Term
    | Term ;

Term: Term '*' Factor
    | Factor ;

Factor: '(' Expr ')'
      | 'INT' ;
