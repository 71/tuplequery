#[cfg(feature = "gat")]
#[macro_export]
macro_rules! gat {
    ( $(#$attr: tt)* type $name: ident < $($arg_name: ident: ($($arg_trait: tt)*)),+ > : $($impl: tt)* ) => {
        $(#$attr)*
        type $name< $($arg_name : $($arg_trait)*),+ > : $($impl)*;
    };

    ( $(#$attr: tt)* type $name: ident < $($arg_name: ident: ($($arg_trait: tt)*)),+ > = $($impl: tt)* ) => {
        $(#$attr)*
        type $name< $($arg_name : $($arg_trait)*),+ > = $($impl)*;
    };

    ( $self: ident :: $name: ident < $($arg_name: ident),+ >) => {
        $self::$name< $($arg_name),+ >
    };
}

#[cfg(not(feature = "gat"))]
#[macro_export]
macro_rules! gat {
    ( $(#$attr: tt)* type $name: ident < $($arg_name: ident: ($($arg_trait: tt)*)),+ > : $($impl: tt)* ) => {
        $(#$attr)*
        type $name: $($impl)*;
    };

    ( $(#$attr: tt)* type $name: ident < $($arg_name: ident: ($($arg_trait: tt)*)),+ > = $($impl: tt)* ) => {
        $(#$attr)*
        type $name = $($impl)*;
    };

    ( $self: ident :: $name: ident < $($arg_name: ident),+ >) => {
        $self::$name
    };
}
