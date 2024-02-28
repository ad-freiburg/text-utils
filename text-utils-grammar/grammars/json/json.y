%start json
%%
json
    : value 
    ;

obj
    : '{' '}' 
    | '{' pair_plus '}' 
    ;

pair_plus
    : pair 
    | pair_plus ',' pair 
    ;

pair
    : 'STRING' ':' value 
    ;

arr
    : '[' ']' 
    | '[' arr_plus ']' 
    ;

arr_plus
    : value 
    | arr_plus ',' value 
    ;

value
    : 'STRING' 
    | 'NUMBER' 
    | obj 
    | arr 
    | 'true' 
    | 'false' 
    | 'null' 
    ;
