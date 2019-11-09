//! A custom lexer for AWK for use with LALRPOP.
//!
//! This lexer is fairly rudamentary. It ought not be too slow, but it also has not been optimized
//! very aggressively. Various portions of the AWK language are not yet supported.
use regex::Regex;
use unicode_xid::UnicodeXID;

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
    Match,
    NotMatch,

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
    ["~", Tok::Match],
    ["!~", Tok::NotMatch],
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
            res[k.len() - 1].push((k.as_bytes(), v.clone()));
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

fn is_id_start(c: char) -> bool {
    c == '_' || c.is_xid_start()
}

fn is_id_body(c: char) -> bool {
    c == '_' || c == '\'' || c.is_xid_continue()
}

impl<'a, 'b> Tokenizer<'a, 'b> {
    fn keyword<'c>(&self) -> Option<(Tok<'c>, usize)> {
        let start = self.cur;
        let remaining = self.text.len() - start;
        for (len, ks) in KEYWORDS_BY_LEN.iter().enumerate().rev() {
            let len = len + 1;
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

    fn num<'c>(&self) -> Option<(Tok<'c>, usize)> {
        lazy_static! {
            static ref INT_PATTERN: Regex = Regex::new(r"^[+-]?\d+").unwrap();
            // Adapted from https://www.regular-expressions.info/floatingpoint.html
            static ref FLOAT_PATTERN: Regex = Regex::new(r"^[-+]?\d*\.\d+([eE][-+]?\d+)?").unwrap();
        };
        let text = &self.text[self.cur..];
        if let Some(f) = FLOAT_PATTERN.captures(text).and_then(|c| c.get(0)) {
            let fs = f.as_str();
            Some((Tok::FLit(fs.parse().unwrap()), fs.len()))
        } else if let Some(i) = INT_PATTERN.captures(text).and_then(|c| c.get(0)) {
            let is = i.as_str();
            Some((Tok::ILit(is.parse().unwrap()), is.len()))
        } else {
            None
        }
    }

    fn push_char(&mut self, c: char) {
        let start = self.buf.len();
        self.buf.resize_with(start + c.len_utf8(), Default::default);
        c.encode_utf8(&mut self.buf[start..]);
    }

    fn ident(&mut self, start: char) -> (&'b str, usize) {
        debug_assert!(is_id_start(start));
        self.buf.clear();
        self.push_char(start);
        let ix = self.text[self.cur..]
            .char_indices()
            .take_while(|(_, c)| is_id_body(*c))
            .last()
            .map(|(ix, _)| self.cur + ix + 1)
            .unwrap_or(self.cur);
        self.buf
            .extend_from_slice(&self.text.as_bytes()[self.cur..ix]);
        let s = self
            .arena
            .alloc_str(std::str::from_utf8(&self.buf[..]).unwrap());
        (s, ix)
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
                Ok((s, self.cur + end))
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
                Ok((s, self.cur + end))
            }
            None => Err(Error {
                location: self.cur,
                desc: "incomplete string literal",
            }),
        }
    }

    fn consume_comment(&mut self) {
        let mut iter = self.text[self.cur..].char_indices();
        if let Some((_, '#')) = iter.next() {
            if let Some((ix, _)) = iter.skip_while(|x| x.1 != '\n').next() {
                self.cur += ix;
            } else {
                self.cur = self.text.len();
            }
        }
    }

    fn consume_ws(&mut self) {
        let mut res = 0;
        for (ix, c) in self.text[self.cur..].char_indices() {
            res = ix;
            if c == '\n' || !c.is_whitespace() {
                break;
            }
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

    fn potential_re(&self) -> bool {
        match &self.prev_tok {
            Some(Tok::Ident(_)) | Some(Tok::StrLit(_)) | Some(Tok::PatLit(_))
            | Some(Tok::ILit(_)) | Some(Tok::FLit(_)) => false,
            _ => true,
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
        macro_rules! try_tok {
            ($e:expr) => {
                match $e {
                    Ok(e) => e,
                    Err(e) => return Some(Err(e)),
                };
            };
        }
        self.advance();
        let span = if let Some((ix, c)) = self.text[self.cur..].char_indices().next() {
            let ix = self.cur + ix;
            match c {
                '"' => {
                    self.cur += 1;
                    let (s, new_start) = try_tok!(self.string_lit());
                    self.cur = new_start;
                    (ix, Tok::StrLit(s), new_start)
                }
                '/' if self.potential_re() => {
                    self.cur += 1;
                    let (re, new_start) = try_tok!(self.regex_lit());
                    self.cur = new_start;
                    (ix, Tok::PatLit(re), new_start)
                }
                c => {
                    if let Some((tok, len)) = self.keyword() {
                        self.cur += len;
                        (ix, tok, self.cur)
                    } else if let Some((tok, len)) = self.num() {
                        self.cur += len;
                        (ix, tok, self.cur)
                    } else if is_id_start(c) {
                        self.cur += c.len_utf8();
                        let (s, new_start) = self.ident(c);
                        self.cur = new_start;
                        (ix, Tok::Ident(s), self.cur)
                    } else {
                        return None;
                    }
                }
            }
        } else {
            return None;
        };
        self.prev_tok = Some(span.1.clone());
        Some(Ok(span))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn lex_str<'b>(a: &'b Arena, s: &str) -> Vec<Spanned<Tok<'b>>> {
        Tokenizer::new(s, a).map(|x| x.ok().unwrap()).collect()
    }

    #[test]
    fn basic() {
        let a = Arena::default();
        let toks = lex_str(
            &a,
            r#" if (x == yzk) {
            print x<y, y<=z, z;
        }"#,
        );
        use Tok::*;
        assert_eq!(
            toks.into_iter().map(|x| x.1).collect::<Vec<_>>(),
            vec![
                If,
                LParen,
                Ident("x"),
                EQ,
                Ident("yzk"),
                RParen,
                LBrace,
                Newline,
                Print,
                Ident("x"),
                LT,
                Ident("y"),
                Comma,
                Ident("y"),
                LTE,
                Ident("z"),
                Comma,
                Ident("z"),
                Semi,
                Newline,
                RBrace
            ]
        );
    }

    #[test]
    fn literals() {
        let a = Arena::default();
        let toks = lex_str(
            &a,
            r#" x="\"hi\tthere\n"; b=/hows it \/going/; x="重庆辣子鸡"; c= 1 / 3.5 "#,
        );
        use Tok::*;
        assert_eq!(
            toks.into_iter().map(|x| x.1).collect::<Vec<_>>(),
            vec![
                Ident("x"),
                Assign,
                StrLit("\"hi\tthere\n"),
                Semi,
                Ident("b"),
                Assign,
                PatLit("hows it /going"),
                Semi,
                Ident("x"),
                Assign,
                StrLit("重庆辣子鸡"),
                Semi,
                Ident("c"),
                Assign,
                ILit(1),
                Div,
                FLit(3.5),
            ],
        )
    }
}
