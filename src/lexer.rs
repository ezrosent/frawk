//! A custom lexer for AWK for use with LALRPOP.
//!
//! This lexer is fairly rudamentary. It ought not be too slow, but it also has not been optimized
//! very aggressively. Various portions of the AWK language are not yet supported.
use regex::Regex;
use unicode_xid::UnicodeXID;

use crate::arena::Arena;

pub type Spanned<T> = (usize, T, usize);

#[derive(Debug, PartialEq, Clone)]
pub enum Tok<'a> {
    Begin,
    End,
    Break,
    Continue,
    Next,
    NextFile,
    For,
    If,
    Else,
    Print,
    Printf,
    // Separate token for a "print(" and "printf(".
    PrintLP,
    PrintfLP,
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
    Not,

    AND,
    OR,
    QUESTION,
    COLON,

    Append, // >>

    Dollar,
    Semi,
    Newline,
    Comma,
    In,
    Delete,
    Function,
    Return,

    Ident(&'a str),
    StrLit(&'a str),
    PatLit(&'a str),
    CallStart(&'a str),

    ILit(&'a str),
    FLit(&'a str),
}

static_map!(
    KEYWORDS<&'static str, Tok<'static>>,
    ["BEGIN", Tok::Begin],
    ["END", Tok::End],
    ["break", Tok::Break],
    ["continue", Tok::Continue],
    ["next", Tok::Next],
    ["nextfile", Tok::NextFile],
    ["for", Tok::For],
    ["if", Tok::If],
    ["else", Tok::Else],
    ["print", Tok::Print],
    ["printf", Tok::Printf],
    ["print(", Tok::PrintLP],
    ["printf(", Tok::PrintfLP],
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
    [",", Tok::Comma],
    ["in", Tok::In],
    ["!", Tok::Not],
    ["&&", Tok::AND],
    ["||", Tok::OR],
    ["?", Tok::QUESTION],
    [":", Tok::COLON],
    ["delete", Tok::Delete],
    ["function", Tok::Function],
    ["return", Tok::Return],
    ["$", Tok::Dollar]
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

pub struct Tokenizer<'a> {
    text: &'a str,
    cur: usize,
    prev_tok: Option<Tok<'a>>,
}

fn is_id_start(c: char) -> bool {
    c == '_' || c.is_xid_start()
}

fn is_id_body(c: char) -> bool {
    c == '_' || c == '\'' || c.is_xid_continue()
}

fn push_char(buf: &mut Vec<u8>, c: char) {
    let start = buf.len();
    buf.resize_with(start + c.len_utf8(), Default::default);
    c.encode_utf8(&mut buf[start..]);
}

pub(crate) fn parse_string_literal<'a, 'outer>(
    lit: &str,
    arena: &'a Arena<'outer>,
    buf: &mut Vec<u8>,
) -> &'a str {
    // assumes we just saw a '"'
    buf.clear();
    let mut is_escape = false;
    for c in lit.chars() {
        if is_escape {
            match c {
                'a' => buf.push(0x07), // BEL
                'b' => buf.push(0x08), // BS
                'f' => buf.push(0x0C), // FF
                'v' => buf.push(0x0B), // VT
                '\\' => buf.push('\\' as u8),
                'n' => buf.push('\n' as u8),
                'r' => buf.push('\r' as u8),
                't' => buf.push('\t' as u8),
                '"' => buf.push('"' as u8),
                c => {
                    buf.push('\\' as u8);
                    push_char(buf, c);
                }
            };
            is_escape = false;
        } else {
            match c {
                '\\' => {
                    is_escape = true;
                    continue;
                }
                c => {
                    push_char(buf, c);
                }
            }
        }
    }
    std::str::from_utf8(arena.alloc_bytes(&buf[..])).unwrap()
}

pub(crate) fn parse_regex_literal<'a, 'outer>(
    lit: &str,
    arena: &'a Arena<'outer>,
    buf: &mut Vec<u8>,
) -> &'a str {
    buf.clear();
    let mut is_escape = false;
    for c in lit.chars() {
        if is_escape {
            match c {
                '/' => buf.push('/' as u8),
                c => {
                    buf.push('\\' as u8);
                    push_char(buf, c);
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
                    break;
                }
                c => {
                    push_char(buf, c);
                }
            }
        }
    }
    std::str::from_utf8(arena.alloc_bytes(&buf[..])).unwrap()
}

impl<'a> Tokenizer<'a> {
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

    fn num(&self) -> Option<(Tok<'a>, usize)> {
        lazy_static! {
            static ref INT_PATTERN: Regex = Regex::new(r"^[+-]?\d+").unwrap();
            // Adapted from https://www.regular-expressions.info/floatingpoint.html
            static ref FLOAT_PATTERN: Regex = Regex::new(r"^[-+]?\d*\.\d+([eE][-+]?\d+)?").unwrap();
        };
        let text = &self.text[self.cur..];
        if let Some(f) = FLOAT_PATTERN.captures(text).and_then(|c| c.get(0)) {
            let fs = f.as_str();
            Some((Tok::FLit(fs), fs.len()))
        } else if let Some(i) = INT_PATTERN.captures(text).and_then(|c| c.get(0)) {
            let is = i.as_str();
            Some((Tok::ILit(is), is.len()))
        } else {
            None
        }
    }

    fn ident(&mut self, id_start: usize) -> (&'a str, usize) {
        debug_assert!(is_id_start(self.text[id_start..].chars().next().unwrap()));
        let ix = self.text[self.cur..]
            .char_indices()
            .take_while(|(_, c)| is_id_body(*c))
            .last()
            .map(|(ix, _)| self.cur + ix + 1)
            .unwrap_or(self.cur);
        (&self.text[id_start..ix], ix)
    }

    fn literal(&mut self, delim: char, error_msg: &'static str) -> Result<(&'a str, usize), Error> {
        // assumes we just saw a delimiter.
        let mut bound = None;
        let mut is_escape = false;
        for (ix, c) in self.text[self.cur..].char_indices() {
            if is_escape {
                is_escape = false;
                continue;
            }
            if c == delim {
                bound = Some(ix);
                break;
            }
            if c == '\\' {
                is_escape = true;
            }
        }
        match bound {
            Some(end) => Ok((&self.text[self.cur..self.cur + end], self.cur + end + 1)),
            None => Err(Error {
                location: self.cur,
                desc: error_msg,
            }),
        }
    }

    fn regex_lit(&mut self) -> Result<(&'a str, usize /* new start */), Error> {
        self.literal('/', "incomplete regex literal")
    }

    fn string_lit(&mut self) -> Result<(&'a str, usize /* new start */), Error> {
        self.literal('"', "incomplete string literal")
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
            | Some(Tok::ILit(_)) | Some(Tok::FLit(_)) | Some(Tok::RParen) => false,
            _ => true,
        }
    }
}

#[derive(Debug)]
pub struct Error {
    pub location: usize,
    pub desc: &'static str,
}

impl<'a> Tokenizer<'a> {
    pub fn new(text: &'a str) -> Tokenizer<'a> {
        Tokenizer {
            text,
            cur: 0,
            prev_tok: None,
        }
    }
}

impl<'a> Iterator for Tokenizer<'a> {
    type Item = Result<Spanned<Tok<'a>>, Error>;
    fn next(&mut self) -> Option<Result<Spanned<Tok<'a>>, Error>> {
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
                        let (s, new_start) = self.ident(ix);
                        if self.text.as_bytes()[new_start] == ('(' as u8) {
                            self.cur = new_start + 1;
                            (ix, Tok::CallStart(s), self.cur)
                        } else {
                            self.cur = new_start;
                            (ix, Tok::Ident(s), self.cur)
                        }
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
    fn lex_str<'b>(s: &'b str) -> Vec<Spanned<Tok<'b>>> {
        Tokenizer::new(s).map(|x| x.ok().unwrap()).collect()
    }

    #[test]
    fn basic() {
        let toks = lex_str(
            r#" if (x == yzk){
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
        let toks =
            lex_str(r#" x="\"hi\tthere\n"; b   =/hows it \/going/; x="重庆辣子鸡"; c= 1 / 3.5 "#);
        use Tok::*;
        let s1 = "\\\"hi\\tthere\\n";
        let s2 = "hows it \\/going";
        assert_eq!(
            toks.into_iter().map(|x| x.1).collect::<Vec<_>>(),
            vec![
                Ident("x"),
                Assign,
                StrLit(s1),
                Semi,
                Ident("b"),
                Assign,
                PatLit(s2),
                Semi,
                Ident("x"),
                Assign,
                StrLit("重庆辣子鸡"),
                Semi,
                Ident("c"),
                Assign,
                ILit("1"),
                Div,
                FLit("3.5"),
            ],
        );
        let mut buf = Vec::new();
        let a = Arena::default();
        assert_eq!(parse_string_literal(s1, &a, &mut buf), "\"hi\tthere\n");
        assert_eq!(parse_regex_literal(s2, &a, &mut buf), "hows it /going");
    }
}
