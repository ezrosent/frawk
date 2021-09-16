//! A custom lexer for AWK for use with LALRPOP.
//!
//! This lexer is fairly rudamentary. It ought not be too slow, but it also has not been optimized
//! very aggressively. Various edge cases still do not work.
use hashbrown::HashMap;
use lazy_static::lazy_static;
use regex::Regex;
use unicode_xid::UnicodeXID;

use crate::arena::Arena;

#[derive(PartialEq, Eq, Clone, Debug, Default)]
pub struct Loc {
    pub line: usize,
    pub col: usize,
    offset: usize,
}

pub type Spanned<T> = (Loc, T, Loc);

#[derive(Debug, PartialEq, Clone)]
pub enum Tok<'a> {
    Begin,
    Prepare,
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
    Pow,
    PowAssign,
    Mod,
    ModAssign,
    Match,
    NotMatch,

    EQ,
    NEQ,
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
    Pipe,

    Append, // >>

    Dollar,
    Semi,
    Newline,
    Comma,
    In,
    Delete,
    Return,

    Ident(&'a str),
    StrLit(&'a str),
    PatLit(&'a str),
    CallStart(&'a str),
    FunDec(&'a str),

    ILit(&'a str),
    HexLit(&'a str),
    FLit(&'a str),
}

macro_rules! kw_inner {
    ($m:expr, $k:expr, $v:expr) => {
        $m.insert(&$k[..], ($v, Default::default()));
    };
    ($m:expr, $k:expr, $v1:expr, $v2:expr) => {
        $m.insert(&$k[..], ($v1, Some($v2)));
    };
}

macro_rules! keyword_map {
    ($name:ident<&'static [u8], ($vty1:ty, $vty2:ty)>, $([$($e:tt)*]),*) => {
        lazy_static::lazy_static! {
            pub(crate) static ref $name: hashbrown::HashMap<&'static [u8],($vty1, $vty2)> = {
                let mut m = hashbrown::HashMap::new();
                $(
                    kw_inner!(m, $($e)*);
                )*
                m
            };
        }
    }
}

lazy_static! {
    static ref WS: Regex = Regex::new(r"^\s").unwrap();
    static ref WS_BRACE: Regex = Regex::new(r"^[\s{}]").unwrap();
    static ref WS_SEMI: Regex = Regex::new(r"^[\s;]").unwrap();
    static ref WS_SEMI_NL: Regex = Regex::new(r"^[\s;\n]").unwrap();
    static ref WS_SEMI_NL_RB: Regex = Regex::new(r"^[\s;\n}]").unwrap();
    static ref WS_SEMI_RPAREN: Regex = Regex::new(r"^[\s;)]").unwrap();
    static ref WS_PAREN: Regex = Regex::new(r"^[\s()]").unwrap();
}

keyword_map!(
    KEYWORDS<&'static [u8], (Tok<'static>, Option<Regex>)>,
    [b"PREPARE", Tok::Prepare],
    [b"BEGIN", Tok::Begin, WS_BRACE.clone()],
    [b"END", Tok::End, WS_BRACE.clone()],
    [b"break", Tok::Break, WS_SEMI.clone()],
    [b"continue", Tok::Continue, WS_SEMI.clone()],
    [b"next", Tok::Next],
    [b"nextfile", Tok::NextFile],
    [b"for", Tok::For, WS_PAREN.clone()],
    [b"if", Tok::If],
    [b"else", Tok::Else],
    [b"print", Tok::Print, WS_SEMI_NL_RB.clone()],
    [b"printf", Tok::Printf, WS_SEMI_NL.clone()],
    [b"print(", Tok::PrintLP],
    [b"printf(", Tok::PrintfLP],
    [b"while", Tok::While, WS_PAREN.clone()],
    [b"do", Tok::Do, WS_BRACE.clone()],
    [b"{", Tok::LBrace],
    [b"}", Tok::RBrace],
    [b"[", Tok::LBrack],
    [b"]", Tok::RBrack],
    [b"(", Tok::LParen],
    [b")", Tok::RParen],
    [b"getline", Tok::Getline, WS_SEMI_RPAREN.clone()],
    [b"|", Tok::Pipe],
    [b"=", Tok::Assign],
    [b"+", Tok::Add],
    [b"+=", Tok::AddAssign],
    [b"-", Tok::Sub],
    [b"-=", Tok::SubAssign],
    [b"*", Tok::Mul],
    [b"*=", Tok::MulAssign],
    [b"/", Tok::Div],
    [b"/=", Tok::DivAssign],
    [b"^", Tok::Pow],
    [b"^=", Tok::PowAssign],
    [b"%", Tok::Mod],
    [b"%=", Tok::ModAssign],
    [b"~", Tok::Match],
    [b"!~", Tok::NotMatch],
    [b"==", Tok::EQ],
    [b"!=", Tok::NEQ],
    [b"<", Tok::LT],
    [b"<=", Tok::LTE],
    [b">", Tok::GT],
    [b"--", Tok::Decr],
    [b"++", Tok::Incr],
    [b">=", Tok::GTE],
    [b">>", Tok::Append],
    [b";", Tok::Semi],
    [b"\n", Tok::Newline],
    [b"\r\n", Tok::Newline],
    [b",", Tok::Comma],
    // XXX: hack "in" must have whitespace after it.
    [b"in ", Tok::In],
    [b"in\t", Tok::In],
    [b"!", Tok::Not],
    [b"&&", Tok::AND],
    [b"||", Tok::OR],
    [b"?", Tok::QUESTION],
    [b":", Tok::COLON],
    [b"delete", Tok::Delete, WS_PAREN.clone()],
    [b"return", Tok::Return, WS_PAREN.clone()],
    [b"$", Tok::Dollar]
);

lazy_static! {
    // TODO: use a regex set for this instead. It'd be a bit faster and it would allow "must
    // follow"-style issues to be handled in a clearer way.
    static ref KEYWORDS_BY_LEN: Vec<HashMap<&'static [u8], Tok<'static>>> = {
        let max_len = KEYWORDS.keys().map(|s| s.len()).max().unwrap();
        let mut res: Vec<HashMap<_, _>> = vec![Default::default(); max_len];
        for (k, (v, _)) in KEYWORDS.iter() {
            res[k.len() - 1].insert(*k, v.clone());
        }
        res
    };
}

pub struct Tokenizer<'a> {
    text: &'a str,
    cur: usize,
    prev_tok: Option<Tok<'a>>,
    lines: Vec<usize>,
}

pub fn is_ident(s: &str) -> bool {
    for (i, c) in s.chars().enumerate() {
        if i == 0 && !is_id_start(c) {
            return false;
        }
        if i > 0 && !is_id_body(c) {
            return false;
        }
    }
    true
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

pub(crate) fn parse_string_literal<'a>(lit: &str, arena: &'a Arena, buf: &mut Vec<u8>) -> &'a [u8] {
    fn hex_digit(c: char) -> Option<u8> {
        match c {
            '0'..='9' => Some((c as u8) - b'0'),
            'a'..='f' => Some(10 + (c as u8) - b'a'),
            'A'..='F' => Some(10 + (c as u8) - b'A'),
            _ => None,
        }
    }
    fn octal_digit(c: char) -> Option<u8> {
        match c {
            '0'..='7' => Some((c as u8) - b'0'),
            _ => None,
        }
    }
    // assumes we just saw a '"'
    buf.clear();
    let mut is_escape = false;
    let mut iter = lit.chars();
    'top: while let Some(mut c) = iter.next() {
        'mid: loop {
            if is_escape {
                match c {
                    'a' => buf.push(0x07), // BEL
                    'b' => buf.push(0x08), // BS
                    'f' => buf.push(0x0C), // FF
                    'v' => buf.push(0x0B), // VT
                    '\\' => buf.push(b'\\'),
                    'n' => buf.push(b'\n'),
                    'r' => buf.push(b'\r'),
                    't' => buf.push(b'\t'),
                    '"' => buf.push(b'"'),
                    // 1 or 2 hex digits
                    'x' => {
                        let mut n = if let Some(x) = iter.next() {
                            if let Some(d) = hex_digit(x) {
                                d
                            } else {
                                // no digits, push \x and repeat.
                                buf.push(b'\\');
                                buf.push(b'x');
                                c = x;
                                is_escape = false;
                                continue;
                            }
                        } else {
                            buf.push(b'\\');
                            buf.push(b'x');
                            break 'top;
                        };
                        if let Some(x) = iter.next() {
                            if let Some(d) = hex_digit(x) {
                                // We cannot overflow
                                n *= 16;
                                n += d;
                            } else {
                                buf.push(n);
                                is_escape = false;
                                c = x;
                                continue;
                            }
                        }
                        buf.push(n)
                    }
                    // 1 to 3 octal digits
                    '0'..='7' => {
                        let mut n = octal_digit(c).unwrap();
                        for _ in 0..2 {
                            if let Some(x) = iter.next() {
                                if let Some(d) = hex_digit(x) {
                                    // saturate on overflow
                                    n = n.saturating_mul(8);
                                    n = n.saturating_add(d);
                                } else {
                                    buf.push(n);
                                    is_escape = false;
                                    c = x;
                                    continue 'mid;
                                }
                            }
                        }
                        buf.push(n);
                    }
                    _ => {
                        buf.push(b'\\');
                        push_char(buf, c);
                    }
                };
                is_escape = false;
            } else {
                match c {
                    '\\' => is_escape = true,
                    c => push_char(buf, c),
                }
            }
            break;
        }
    }
    arena.alloc_bytes(&buf[..])
}

pub(crate) fn parse_regex_literal<'a>(lit: &str, arena: &'a Arena, buf: &mut Vec<u8>) -> &'a [u8] {
    // Regexes have their own escaping rules, let them apply them. The only think we look to do is
    // replace "\/" with "/".
    // NB: Awk escaping rules are a subset of Rust's regex escape rule syntax, but if we applied
    // Awk's rewrites here, we might create a pattern that is invalid UTF-8, which will cause a
    // failure when we try and compile the regular expression.
    buf.clear();
    let mut is_escape = false;
    for c in lit.chars() {
        if is_escape {
            match c {
                '/' => buf.push(b'/'),
                c => {
                    buf.push(b'\\');
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
    arena.alloc_bytes(&buf[..])
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
            let text = &self.text.as_bytes()[start..start + len];
            if let Some(tok) = ks.get(text) {
                if start + len == self.text.len()
                    || KEYWORDS
                        .get(text)
                        .unwrap()
                        .1
                        .as_ref()
                        .map_or(true, |x| x.is_match(&self.text[start + len..]))
                {
                    return Some((tok.clone(), len));
                }
            }
        }
        None
    }

    fn num(&self) -> Option<(Tok<'a>, usize)> {
        lazy_static! {
            static ref HEX_PATTERN: Regex = Regex::new(r"^[+-]?0[xX][0-9A-Fa-f]+").unwrap();
            static ref INT_PATTERN: Regex = Regex::new(r"^[+-]?\d+").unwrap();
            // Adapted from https://www.regular-expressions.info/floatingpoint.html
            static ref FLOAT_PATTERN: Regex = Regex::new(r"^[-+]?\d*\.\d+([eE][-+]?\d+)?").unwrap();
        };
        let text = &self.text[self.cur..];
        if let Some(i) = HEX_PATTERN.captures(text).and_then(|c| c.get(0)) {
            let is = i.as_str();
            return Some((Tok::HexLit(is), is.len()));
        } else if let Some(f) = FLOAT_PATTERN.captures(text).and_then(|c| c.get(0)) {
            let fs = f.as_str();
            Some((Tok::FLit(fs), fs.len()))
        } else if let Some(i) = INT_PATTERN.captures(text).and_then(|c| c.get(0)) {
            let is = i.as_str();
            Some((Tok::ILit(is), is.len()))
        } else {
            None
        }
    }

    fn fundec(&self) -> Option<(Tok<'a>, usize)> {
        lazy_static! {
            static ref FN_PATTERN: Regex =
                Regex::new(r"^(function\s+([a-zA-Z_][a-zA-Z_0-9]*))\(").unwrap();
        }
        let captures = FN_PATTERN.captures(&self.text[self.cur..])?;
        let full = captures.get(1)?.as_str();
        let name = captures.get(2)?.as_str();
        if KEYWORDS.get(name.as_bytes()).is_none() {
            Some((Tok::FunDec(name), full.len()))
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
                location: self.index_to_loc(self.cur),
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
        let mut iter = self.text[self.cur..].char_indices();
        'outer: while let Some((ix, c)) = iter.next() {
            loop {
                res = ix;
                if c == '\\' {
                    // look ahead for a newline and hence a line continuation
                    if let Some((_, next_c)) = iter.next() {
                        if next_c == '\n' {
                            // count this as whitespace
                            continue 'outer;
                        }
                        break 'outer;
                    }
                }
                if c == '\n' || !c.is_whitespace() {
                    break 'outer;
                }

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
            | Some(Tok::ILit(_)) | Some(Tok::FLit(_)) | Some(Tok::RParen) | Some(Tok::RBrack) => {
                false
            }
            _ => true,
        }
    }
}

#[derive(Debug)]
pub struct Error {
    pub location: Loc,
    pub desc: &'static str,
}

impl From<&'static str> for Error {
    fn from(s: &'static str) -> Error {
        Error {
            location: Default::default(),
            desc: s,
        }
    }
}

impl<'a> Tokenizer<'a> {
    pub fn new(text: &'a str) -> Tokenizer<'a> {
        Tokenizer {
            // A hack to get around some programs failing to parse due to a trailing newline
            text: text.trim_end_matches('\n'),
            cur: 0,
            prev_tok: None,
            lines: text
                .as_bytes()
                .iter()
                .enumerate()
                .flat_map(|(i, b)| if *b == b'\n' { Some(i) } else { None }.into_iter())
                .collect(),
        }
    }
    fn index_to_loc(&self, ix: usize) -> Loc {
        let offset = ix;
        match self.lines.binary_search(&ix) {
            Ok(0) | Err(0) => Loc {
                line: 0,
                col: ix,
                offset,
            },
            Ok(line) => Loc {
                line: line - 1,
                col: ix - self.lines[line - 1] - 1,
                offset,
            },
            Err(line) => Loc {
                line,
                col: ix - self.lines[line - 1] - 1,
                offset,
            },
        }
    }
    fn spanned<T>(&self, l: usize, r: usize, t: T) -> Spanned<T> {
        (self.index_to_loc(l), t, self.index_to_loc(r))
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
                }
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
                    self.spanned(ix, new_start, Tok::StrLit(s))
                }
                '/' if self.potential_re() => {
                    self.cur += 1;
                    let (re, new_start) = try_tok!(self.regex_lit());
                    self.cur = new_start;
                    self.spanned(ix, new_start, Tok::PatLit(re))
                }
                c => {
                    if let Some((tok, len)) = self.fundec() {
                        self.cur += len;
                        self.spanned(ix, self.cur, tok)
                    } else if let Some((tok, len)) = self.keyword() {
                        self.cur += len;
                        self.spanned(ix, self.cur, tok)
                    } else if let Some((tok, len)) = self.num() {
                        self.cur += len;
                        self.spanned(ix, self.cur, tok)
                    } else if is_id_start(c) {
                        self.cur += c.len_utf8();
                        let (s, new_start) = self.ident(ix);
                        let bs = self.text.as_bytes();
                        if new_start < bs.len() && self.text.as_bytes()[new_start] == b'(' {
                            self.cur = new_start + 1;
                            self.spanned(ix, self.cur, Tok::CallStart(s))
                        } else {
                            self.cur = new_start;
                            self.spanned(ix, self.cur, Tok::Ident(s))
                        }
                    } else {
                        return None;
                    }
                }
            }
        } else if let Some(Tok::Newline) = self.prev_tok {
            return None;
        } else {
            self.spanned(self.cur, self.cur, Tok::Newline)
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
    fn locations() {
        const TEXT: &'static str = r#"This is the first line
and the second
and the third"#;
        let tok = Tokenizer::new(TEXT);
        assert_eq!(
            tok.index_to_loc(4),
            Loc {
                line: 0,
                col: 4,
                offset: 4,
            }
        );
        assert_eq!(
            tok.index_to_loc(22),
            Loc {
                line: 0,
                col: 22,
                offset: 22,
            }
        );
        assert_eq!(
            tok.index_to_loc(23),
            Loc {
                line: 1,
                col: 0,
                offset: 23,
            }
        );
        let tok2 = Tokenizer::new("\nhello");
        assert_eq!(
            tok2.index_to_loc(0),
            Loc {
                line: 0,
                col: 0,
                offset: 0
            },
        );
        assert_eq!(
            tok2.index_to_loc(1),
            Loc {
                line: 1,
                col: 0,
                offset: 1
            },
        );
        assert_eq!(
            tok2.index_to_loc(2),
            Loc {
                line: 1,
                col: 1,
                offset: 2
            },
        );
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
                RBrace,
                Newline,
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
        assert_eq!(parse_string_literal(s1, &a, &mut buf), b"\"hi\tthere\n");
        assert_eq!(parse_regex_literal(s2, &a, &mut buf), b"hows it /going");
        assert_eq!(
            parse_string_literal(r#"are you there \77\x3f"#, &a, &mut buf),
            b"are you there ??"
        );
        assert_eq!(
            parse_string_literal(r#"are you there \77\x"#, &a, &mut buf),
            b"are you there ?\\x"
        );
        assert_eq!(
            parse_string_literal(r#"are you there \77\xh"#, &a, &mut buf),
            b"are you there ?\\xh"
        );
    }
}
