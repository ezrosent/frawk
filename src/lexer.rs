use crate::runtime::{Float, Int};
use std::str::CharIndices;

pub type Spanned<T> = (usize, T, usize);

pub enum Tok<'a> {
    Begin,
    End,
    Break,
    Continue,
    For,
    If,
    Print,
    While,
    Do,

    // { }
    LBrace,
    RBrace,
    // [ ]
    LBrack,
    RBrack,
    // ( )
    LParen,
    RParen,

    Getline,
    Assign,
    Add,
    AddAssign,
    Sub,
    SubAssign,
    Mul,
    MulAssign,
    Div,
    DivAssign,
    Mod,
    ModAssign,

    EQ,
    LT,
    GT,
    LTE,
    GTE,
    Incr,
    Decr,

    Append, // >>

    Semi,
    Newline,
    Comma,

    Ident(&'a str),
    StrLit(&'a str),
    PatLit(&'a str),

    ILit(Int),
    FLit(Float),
}

static_map!(
    KEYWORDS<&'static str, Tok<'static>>,
    ["BEGIN", Tok::Begin],
    ["END", Tok::End],
    ["break", Tok::Break],
    ["continue", Tok::Continue],
    ["for", Tok::For],
    ["if", Tok::If],
    ["print", Tok::Print],
    ["while", Tok::While],
    ["do", Tok::Do],
    ["{", Tok::LBrace],
    ["}", Tok::RBrace],
    ["[", Tok::LBrack],
    ["]", Tok::RBrack],
    ["(", Tok::LParen],
    [")", Tok::RParen],
    ["getline", Tok::Getline],
    ["=", Tok::Assign],
    ["+", Tok::Add],
    ["+=", Tok::AddAssign],
    ["-", Tok::Sub],
    ["-=", Tok::SubAssign],
    ["*", Tok::Mul],
    ["*=", Tok::MulAssign],
    ["/", Tok::Div],
    ["/=", Tok::DivAssign],
    ["%", Tok::Mod],
    ["%=", Tok::ModAssign],
    ["==", Tok::EQ],
    ["<", Tok::LT],
    ["<=", Tok::LTE],
    [">", Tok::GT],
    ["--", Tok::Decr],
    ["++", Tok::Incr],
    [">=", Tok::GTE],
    [">>", Tok::Append],
    [";", Tok::Semi],
    [",", Tok::Comma]
);
static_map!(
    SINGLE_CHAR_KEYWORDS<&'static str, Tok<'static>>,
    ["BEGIN", Tok::Begin],
    ["END", Tok::End],
    ["break", Tok::Break],
    ["continue", Tok::Continue],
    ["for", Tok::For],
    ["if", Tok::If],
    ["print", Tok::Print],
    ["while", Tok::While],
    ["do", Tok::Do],
    ["{", Tok::LBrace],
    ["}", Tok::RBrace],
    ["[", Tok::LBrack],
    ["]", Tok::RBrack],
    ["(", Tok::LParen],
    [")", Tok::RParen],
    ["getline", Tok::Getline],
    ["+", Tok::Add],
    ["+=", Tok::AddAssign],
    ["-", Tok::Sub],
    ["-=", Tok::SubAssign],
    ["*", Tok::Mul],
    ["*=", Tok::MulAssign],
    ["/", Tok::Div],
    ["/=", Tok::DivAssign],
    ["%", Tok::Mod],
    ["%=", Tok::ModAssign],
    ["==", Tok::EQ],
    ["<", Tok::LT],
    ["<=", Tok::LTE],
    [">", Tok::GT],
    ["--", Tok::Decr],
    ["++", Tok::Incr],
    [">=", Tok::GTE],
    [">>", Tok::Append],
    [";", Tok::Semi],
    ["\n", Tok::Semi],
    [",", Tok::Comma]
);

pub struct Tokenizer<'a> {
    text: &'a str,
    chars: CharIndices<'a>,
}

pub struct Error {
    pub location: usize,
    pub desc: &'static str,
}

impl<'a> Tokenizer<'a> {
    pub fn new(text: &'a str) -> Tokenizer<'a> {
        Tokenizer {
            text,
            chars: text.char_indices(),
        }
    }
}

impl<'a> Iterator for Tokenizer<'a> {
    type Item = Result<Spanned<Tok<'a>>, Error>;
    fn next(&mut self) -> Option<Result<Spanned<Tok<'a>>, Error>> {
        loop {
            let (ix, c) = match self.chars.next() {
                Some(x) => x,
                None => return None,
            };
            // XXX what about newlines?
            if c.is_whitespace() {
                continue;
            }
            // TODO match any keywords and emit a token. We can probably write a generic function
            //      for this that looks ahead N bytes into the string and checks it.
            // TODO add support for allocating strings in arenas, we will have to build up a
            //      transformed string for string literals (due to escape sequences), we can add a
            //      buffer to Tokenizer to hold those. That should keep allocations low.
            // TODO add arena to Tokenizer.
            // TODO implement literal parsing. String literals will be tricky but should be easy
            //      enough. But what about disambiguating division from regexes? interestingly
            //      enough nawk handles these at parse time, but it requires a semantic action
            //      after parsing a '/', which is interesting...
            //
            //      I think we can just special-case '/' based on preceeding tokens, at least for
            //      now.
            //      If last token is '(', '=' '~', or it is the beginning of a line (a la
            //      patterns?) then it is the start of a regular expression. Otherwise it is a DIV.
            // TODO for brackets, do we need to enforce matching, or can that happen at parse time.
        }
    }
}
