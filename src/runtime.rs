use crate::common::Result;
use hashbrown::HashMap;
use regex::Regex;
use smallvec::SmallVec;
use std::cell::{Ref, RefCell};
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::marker::PhantomData;
use std::rc::Rc;
pub(crate) trait Scalar {}
impl Scalar for Int {}
impl Scalar for Float {}
impl<'a> Scalar for Str<'a> {}

#[derive(Clone, Debug)]
enum Inner<'a> {
    Literal(&'a str),
    Boxed(Rc<str>),
    Concat(Rc<Branch<'a>>),
}

#[derive(Clone, Debug)]
struct Branch<'a> {
    len: u32,
    left: Str<'a>,
    right: Str<'a>,
}

#[derive(Clone, Debug)]
pub(crate) struct Str<'a>(RefCell<Inner<'a>>);

impl<'a> Str<'a> {
    fn len_u32(&self) -> u32 {
        use Inner::*;
        match &*self.0.borrow() {
            Literal(s) => conv_len(s.len()),
            Boxed(s) => conv_len(s.len()),
            Concat(b) => b.len,
        }
    }
    pub(crate) fn len(&self) -> usize {
        use Inner::*;
        match &*self.0.borrow() {
            Literal(s) => s.len(),
            Boxed(s) => s.len(),
            Concat(b) => b.len as usize,
        }
    }
}

impl<'a> PartialEq for Str<'a> {
    fn eq(&self, other: &Str<'a>) -> bool {
        use Inner::*;
        if self.len() != other.len() {
            return false;
        }
        match (&*self.0.borrow(), &*other.0.borrow()) {
            (Literal(s1), Literal(s2)) => return s1 == s2,
            (Boxed(s1), Boxed(s2)) => return s1 == s2,
            (Literal(r), Boxed(b)) | (Boxed(b), Literal(r)) => return *r == &**b,
            (_, _) => {}
        }
        self.force();
        other.force();
        self == other
    }
}

fn conv_len(l: usize) -> u32 {
    if l > (u32::max_value() as usize) {
        u32::max_value()
    } else {
        l as u32
    }
}

impl<'a> Eq for Str<'a> {}

impl<'a> From<&'a str> for Str<'a> {
    fn from(s: &'a str) -> Str<'a> {
        Str(RefCell::new(Inner::Literal(s)))
    }
}
impl<'a> From<String> for Str<'a> {
    fn from(s: String) -> Str<'a> {
        Str(RefCell::new(Inner::Boxed(s.into())))
    }
}

impl<'a> Str<'a> {
    pub(crate) fn clone_str(&self) -> Rc<str> {
        self.force();
        match &*self.0.borrow() {
            Inner::Literal(l) => (*l).into(),
            Inner::Boxed(b) => b.clone(),
            _ => unreachable!(),
        }
    }
    pub(crate) fn with_str<R>(&self, f: impl FnOnce(&str) -> R) -> R {
        self.force();
        match &*self.0.borrow() {
            Inner::Literal(l) => f(l),
            Inner::Boxed(b) => f(&*b),
            _ => unreachable!(),
        }
    }
    pub(crate) fn concat(s1: Str<'a>, s2: Str<'a>) -> Self {
        Str(RefCell::new(Inner::Concat(Rc::new(Branch {
            len: s1.len_u32().saturating_add(s2.len_u32()),
            left: s1,
            right: s2,
        }))))
    }
    /// force flattens the string by concatenating all components into a single boxed string.
    fn force(&self) {
        use Inner::*;
        if let Literal(_) | Boxed(_) = &*self.0.borrow() {
            return;
        }
        let mut cur = self.clone();
        let mut res = String::with_capacity(self.len());
        let mut todos = SmallVec::<[Str<'a>; 16]>::new();
        loop {
            cur = loop {
                match &*cur.0.borrow() {
                    Literal(s) => res.push_str(s),
                    Boxed(s) => res.push_str(&*s),
                    Concat(rc) => {
                        todos.push(rc.right.clone());
                        break rc.left.clone();
                    }
                }
                if let Some(c) = todos.pop() {
                    break c;
                }
                self.0.replace(Boxed(res.into()));
                return;
            };
        }
    }
}

#[cfg(test)]
mod string_tests {
    use super::*;
    #[test]
    fn concat_test() {
        let s1 = Str::from("hi there fellow");
        let s2 = Str::concat(
            Str::concat(Str::from("hi"), Str::from(String::from(" there"))),
            Str::concat(
                Str::from(" "),
                Str::concat(Str::from("fel"), Str::from("low")),
            ),
        );
        assert_eq!(s1, s2);
        assert!(s1 != Str::concat(s1.clone(), s2));
    }
}

#[derive(Default)]
pub(crate) struct RegexCache(Registry<Regex>);

impl RegexCache {
    pub(crate) fn match_regex(&mut self, pat: &Str, s: &Str) -> Result<bool> {
        self.0.get(
            pat,
            |s| match Regex::new(s) {
                Ok(r) => Ok(r),
                Err(e) => err!("{}", e),
            },
            |re| s.with_str(|raw| re.is_match(raw)),
        )
    }
}

#[derive(Default)]
pub(crate) struct FileWrite(Registry<io::BufWriter<File>>);
impl FileWrite {
    pub(crate) fn write_str(&mut self, path: &Str, s: &Str, append: bool) -> Result<()> {
        self.0.get_fallible(
            path,
            |s| match std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .append(append)
                .open(s)
            {
                Ok(f) => Ok(BufWriter::new(f)),
                Err(e) => err!(
                    "failed to open file {} for {}: {}",
                    s,
                    if append { "append" } else { "write" },
                    e
                ),
            },
            |writer| {
                if let Err(e) = s.with_str(|s| writer.write_all(s.as_bytes())) {
                    err!("failed to write to file: {}", e)
                } else {
                    Ok(())
                }
            },
        )
    }
}

#[derive(Default)]
pub(crate) struct FileRead(Registry<io::BufReader<File>>);

impl FileRead {
    pub(crate) fn get_line(
        &mut self,
        path: &Str,
        into: &mut String,
    ) -> Result<bool /* false = EOF */> {
        self.0.get_fallible(
            path,
            |s| match File::open(s) {
                Ok(f) => Ok(BufReader::new(f)),
                Err(e) => err!("failed to open file '{}': {}", s, e),
            },
            |reader| match reader.read_line(into) {
                Ok(n) => Ok(n > 0),
                Err(e) => err!("{}", e),
            },
        )
    }
}

pub(crate) struct Registry<T> {
    // TODO(ezr): we could potentially increase speed here if we did pointer equality (and
    // length) for lookups.
    // We could be fine having duplicates for Regex. We could also also intern strings
    // as we go by swapping out one Rc for another as we encounter them. That would keep the
    // fast path fast, but we would have to make sure we weren't keeping any Refs alive.
    cached: HashMap<Rc<str>, T>,
}
impl<T> Default for Registry<T> {
    fn default() -> Self {
        Registry {
            cached: Default::default(),
        }
    }
}

impl<T> Registry<T> {
    fn get<R>(
        &mut self,
        s: &Str,
        new: impl FnMut(&str) -> Result<T>,
        getter: impl FnOnce(&mut T) -> R,
    ) -> Result<R> {
        self.get_fallible(s, new, |t| Ok(getter(t)))
    }
    fn get_fallible<R>(
        &mut self,
        s: &Str,
        mut new: impl FnMut(&str) -> Result<T>,
        getter: impl FnOnce(&mut T) -> Result<R>,
    ) -> Result<R> {
        use hashbrown::hash_map::Entry;
        let k_str = s.clone_str();
        match self.cached.entry(k_str) {
            Entry::Occupied(mut o) => getter(o.get_mut()),
            Entry::Vacant(v) => {
                let raw_str = &*v.key();
                let mut val = new(raw_str)?;
                let res = getter(&mut val);
                v.insert(val);
                res
            }
        }
    }
}

pub(crate) trait Convert<S, T> {
    fn convert(s: S) -> T;
}

pub(crate) struct _Carrier;

impl Convert<Float, Int> for _Carrier {
    fn convert(f: Float) -> Int {
        f as Int
    }
}
impl Convert<Int, Float> for _Carrier {
    fn convert(i: Int) -> Float {
        i as Float
    }
}
impl<'a> Convert<Int, Str<'a>> for _Carrier {
    fn convert(i: Int) -> Str<'a> {
        format!("{}", i).into()
    }
}
impl<'a> Convert<Float, Str<'a>> for _Carrier {
    fn convert(f: Float) -> Str<'a> {
        let mut buffer = ryu::Buffer::new();
        let printed = buffer.format(f);
        let p_str: String = printed.into();
        p_str.into()
    }
}
impl<'a> Convert<Str<'a>, Float> for _Carrier {
    fn convert(s: Str<'a>) -> Float {
        s.with_str(crate::strton::strtod)
    }
}
impl<'a> Convert<Str<'a>, Int> for _Carrier {
    fn convert(s: Str<'a>) -> Int {
        s.with_str(crate::strton::strtoi)
    }
}
impl<'b, 'a> Convert<&'b Str<'a>, Float> for _Carrier {
    fn convert(s: &'b Str<'a>) -> Float {
        s.with_str(crate::strton::strtod)
    }
}
impl<'b, 'a> Convert<&'b Str<'a>, Int> for _Carrier {
    fn convert(s: &'b Str<'a>) -> Int {
        s.with_str(crate::strton::strtoi)
    }
}

pub(crate) fn convert<S, T>(s: S) -> T
where
    _Carrier: Convert<S, T>,
{
    _Carrier::convert(s)
}

pub(crate) type Int = i64;
pub(crate) type Float = f64;
pub(crate) type IntMap<V> = HashMap<Int, V>;
pub(crate) type StrMap<'a, V> = HashMap<Str<'a>, V>;
pub(crate) struct Iter<S: Scalar>(PhantomData<*const S>);
