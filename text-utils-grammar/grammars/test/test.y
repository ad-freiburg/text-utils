%start start

%%

start: statement ;

statement: statement 'KEYWORD'
     | 'KEYWORD' ;
