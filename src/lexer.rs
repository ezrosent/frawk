use regex::Regex;

use crate::arena::Arena;
use crate::runtime::{Float, Int};

pub type Spanned<T> = (usize, T, usize);

#[derive(Debug, PartialEq, Clone)]
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
    ["\n", Tok::Newline],
    ["\r\n", Tok::Newline],
    [",", Tok::Comma]
);

use lazy_static::lazy_static;

lazy_static! {
    static ref KEYWORDS_BY_LEN: Vec<Vec<(&'static [u8], Tok<'static>)>> = {
        let max_len = KEYWORDS.keys().map(|s| s.len()).max().unwrap();
        let mut res: Vec<Vec<_>> = vec![Default::default(); max_len];
        for (k, v) in KEYWORDS.iter() {
            res[k.len()].push((k.as_bytes(), v.clone()));
        }
        res
    };
}

pub struct Tokenizer<'a, 'b> {
    text: &'a str,
    cur: usize,
    prev_tok: Option<Tok<'b>>,
    buf: Vec<u8>,
    arena: &'b Arena<'b>,
}

impl<'a, 'b> Tokenizer<'a, 'b> {
    fn keyword<'c>(&self) -> Option<(Tok<'c>, usize)> {
        let start = self.cur;
        let remaining = self.text.len() - start;
        for (len, ks) in KEYWORDS_BY_LEN.iter().enumerate().rev() {
            if remaining < len {
                continue;
            }
            for (bs, tok) in ks.iter() {
                debug_assert_eq!(bs.len(), len);
                if *bs == &self.text.as_bytes()[start..start + bs.len()] {
                    return Some((tok.clone(), len));
                }
            }
        }
        None
    }

    fn push_char(&mut self, c: char) {
        let start = self.buf.len();
        self.buf.resize_with(start + c.len_utf8(), Default::default);
        c.encode_utf8(&mut self.buf[start..]);
    }

    fn regex_lit(&mut self) -> Result<(&'b str, usize /* new start */), Error> {
        // assumes we just saw a '/' meeting the disambiguation requirements re: division.
        self.buf.clear();
        let mut bound = None;
        let mut is_escape = false;
        for (ix, c) in self.text[self.cur..].char_indices() {
            if is_escape {
                match c {
                    '/' => self.buf.push('/' as u8),
                    c => {
                        self.buf.push('\\' as u8);
                        self.push_char(c);
                    }
                };
                is_escape = false;
            } else {
                match c {
                    '\\' => {
                        is_escape = true;
                        continue;
                    }
                    '/' => {
                        bound = Some(ix + 1);
                        break;
                    }
                    c => {
                        self.push_char(c);
                    }
                }
            }
        }
        match bound {
            Some(end) => {
                let s = self
                    .arena
                    .alloc_str(std::str::from_utf8(&self.buf[..]).unwrap());
                Ok((s, end))
            }
            None => Err(Error {
                location: self.cur,
                desc: "incomplete regex literal",
            }),
        }
    }

    fn string_lit(&mut self) -> Result<(&'b str, usize /* new start */), Error> {
        // assumes we just saw a '"'
        self.buf.clear();
        let mut bound = None;
        let mut is_escape = false;
        for (ix, c) in self.text[self.cur..].char_indices() {
            if is_escape {
                match c {
                    'a' => self.buf.push(0x07), // BEL
                    'b' => self.buf.push(0x08), // BS
                    'f' => self.buf.push(0x0C), // FF
                    'v' => self.buf.push(0x0B), // VT
                    '\\' => self.buf.push('\\' as u8),
                    'n' => self.buf.push('\n' as u8),
                    'r' => self.buf.push('\r' as u8),
                    't' => self.buf.push('\t' as u8),
                    '"' => self.buf.push('"' as u8),
                    c => {
                        self.buf.push('\\' as u8);
                        self.push_char(c);
                    }
                };
                is_escape = false;
            } else {
                match c {
                    '\\' => {
                        is_escape = true;
                        continue;
                    }
                    '"' => {
                        bound = Some(ix + 1);
                        break;
                    }
                    c => {
                        self.push_char(c);
                    }
                }
            }
        }
        match bound {
            Some(end) => {
                let s = self
                    .arena
                    .alloc_str(std::str::from_utf8(&self.buf[..]).unwrap());
                Ok((s, end))
            }
            None => Err(Error {
                location: self.cur,
                desc: "incomplete string literal",
            }),
        }
    }

    fn consume_comment(&mut self) {
        // assumes we just saw a '#'
        if let Some((ix, _)) = self.text[self.cur..]
            .char_indices()
            .skip_while(|x| x.1 != '\n')
            .next()
        {
            self.cur += ix;
        } else {
            self.cur = self.text.len();
        }
    }

    fn consume_ws(&mut self) {
        let mut res = 0;
        for (ix, c) in self.text[self.cur..].char_indices() {
            if c == '\n' || !c.is_whitespace() {
                break;
            }
            res = ix;
        }
        self.cur += res;
    }

    fn advance(&mut self) {
        let mut prev = self.cur;
        loop {
            self.consume_ws();
            self.consume_comment();
            if self.cur == prev {
                break;
            }
            prev = self.cur;
        }
    }
}

pub struct Error {
    pub location: usize,
    pub desc: &'static str,
}

impl<'a, 'b> Tokenizer<'a, 'b> {
    pub fn new(text: &'a str, arena: &'b Arena) -> Tokenizer<'a, 'b> {
        Tokenizer {
            text,
            arena,
            cur: 0,
            prev_tok: None,
            buf: Default::default(),
        }
    }
}

impl<'a, 'b> Iterator for Tokenizer<'a, 'b> {
    type Item = Result<Spanned<Tok<'b>>, Error>;
    fn next(&mut self) -> Option<Result<Spanned<Tok<'b>>, Error>> {
        loop {
            // psuedocode:
            // repeatedly
            //   advance
            //   match first char:
            //    * '"' => string_lit
            //    * '/' when last token was ;,\n,(, => regex
            //    * _ => number | keyword | identifier
            //
        }
    }
}
