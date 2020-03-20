/// Custom string implemenation.
///
/// There is a lot of unsafe code here. Many of the features here can and were implementable in
/// terms of safe code using enums, and various components of the standard library. We moved to
/// this representation because it significanly improved some benchmarks in terms of time and
/// space, and it also makes for more ergonomic interop with LLVM.
///
/// TODO explain more about what is going on here.
use crate::runtime::{Float, Int};

use regex::Regex;
use smallvec::SmallVec;

use std::alloc::{alloc_zeroed, dealloc, realloc, Layout};
use std::cell::{Cell, UnsafeCell};
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::marker::PhantomData;
use std::mem;
use std::ptr;
use std::rc::Rc;
use std::slice;
use std::str;

// TODO look into a design based on unions

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(usize)]
enum StrTag {
    // TODO we probably do not need Empty, now that we have Inline
    Empty = 0,
    Literal = 1,
    Shared = 2,
    Concat = 3,
    Boxed = 4,
    Inline = 5,
}
const NUM_VARIANTS: usize = 6;

impl StrTag {
    fn forced(self) -> bool {
        use StrTag::*;
        match self {
            Empty | Literal | Boxed | Inline => true,
            Concat | Shared => false,
        }
    }
}

// Why the repr(C)? We may rely on the lengths coming first.

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(transparent)]
struct Inline(u128);
const MAX_INLINE_SIZE: usize = 15;

impl Inline {
    unsafe fn from_raw(ptr: *const u8, len: usize) -> Inline {
        debug_assert!(len <= MAX_INLINE_SIZE);
        let mut res = ((len << 3) | StrTag::Inline as usize) as u128;
        ptr::copy_nonoverlapping(
            ptr,
            mem::transmute::<&mut u128, *mut u8>(&mut res).offset(1),
            len,
        );
        Inline(res)
    }
    unsafe fn from_unchecked(bs: &[u8]) -> Inline {
        Self::from_raw(bs.as_ptr(), bs.len())
    }
    fn len(&self) -> usize {
        (self.0 as usize & 0xFF) >> 3
    }
    fn bytes(&self) -> &[u8] {
        unsafe {
            slice::from_raw_parts(
                mem::transmute::<&Inline, *const u8>(self).offset(1),
                self.len(),
            )
        }
    }
}

#[derive(Clone)]
#[repr(C)]
struct Literal<'a> {
    ptr: *const u8,
    len: u64,
    _marker: PhantomData<&'a ()>,
}

#[derive(Clone, Debug)]
#[repr(C)]
struct Boxed {
    buf: Buf,
    len: u64,
}

#[derive(Clone, Debug)]
#[repr(C)]
struct Shared {
    buf: Buf,
    start: u32,
    end: u32,
}

#[derive(Clone, Debug)]
struct ConcatInner<'a> {
    left: Str<'a>,
    right: Str<'a>,
}
#[derive(Clone, Debug)]
#[repr(C)]
struct Concat<'a> {
    inner: Rc<ConcatInner<'a>>,
    len: u64,
}

#[derive(Default, PartialEq, Eq)]
#[repr(C)]
struct StrRep<'a> {
    hi: usize,
    low: u64,
    _marker: PhantomData<&'a ()>,
}

impl<'a> StrRep<'a> {
    fn get_tag(&self) -> StrTag {
        use StrTag::*;
        let tag = self.hi & 0x7;
        assert!(tag < NUM_VARIANTS);
        match tag {
            0 => Empty,
            1 => Literal,
            2 => Shared,
            3 => Concat,
            4 => Boxed,
            5 => Inline,
            _ => unreachable!(),
        }
    }
}

macro_rules! impl_tagged_from {
    ($from:ty, $tag:expr) => {
        impl<'a> From<$from> for StrRep<'a> {
            fn from(s: $from) -> StrRep<'a> {
                let mut rep = unsafe { mem::transmute::<$from, StrRep>(s) };
                rep.hi |= ($tag as usize);
                rep
            }
        }
    };
}

impl_tagged_from!(Shared, StrTag::Shared);
impl_tagged_from!(Concat<'a>, StrTag::Concat);
impl_tagged_from!(Boxed, StrTag::Boxed);
impl_tagged_from!(Inline, StrTag::Inline);

impl<'a> From<Literal<'a>> for StrRep<'a> {
    fn from(s: Literal<'a>) -> StrRep<'a> {
        if s.len <= MAX_INLINE_SIZE as u64 {
            unsafe { Inline::from_raw(s.ptr, s.len as usize).into() }
        } else if s.ptr as usize & 0x7 == 0 {
            let mut rep = unsafe { mem::transmute::<Literal<'a>, StrRep>(s) };
            rep.hi |= StrTag::Literal as usize;
            rep
        } else {
            let buf = unsafe { Buf::read_from_raw(s.ptr, s.len as usize) };
            Boxed { len: s.len, buf }.into()
        }
    }
}

impl<'a> StrRep<'a> {
    fn len(&mut self) -> usize {
        match self.get_tag() {
            StrTag::Empty => 0,
            StrTag::Boxed | StrTag::Literal | StrTag::Concat => self.low as usize,
            StrTag::Shared => unsafe {
                self.view_as(|s: &Shared| s.end as usize - s.start as usize)
            },
            StrTag::Inline => unsafe { self.view_as(|i: &Inline| i.len()) },
        }
    }
    unsafe fn view_as<T, R>(&mut self, f: impl FnOnce(&T) -> R) -> R {
        let old = self.hi;
        self.hi = old & !0x7;
        let res = f(mem::transmute::<&mut StrRep<'a>, &T>(self));
        self.hi = old;
        res
    }
    unsafe fn drop_as<T>(&mut self) {
        let old = self.hi;
        self.hi = old & !0x7;
        ptr::drop_in_place(mem::transmute::<&mut StrRep<'a>, *mut T>(self));
        self.hi = old;
    }

    unsafe fn copy(&self) -> StrRep<'a> {
        StrRep {
            low: self.low,
            hi: self.hi,
            _marker: PhantomData,
        }
    }
}

impl<'a> Drop for StrRep<'a> {
    fn drop(&mut self) {
        let tag = self.get_tag();
        unsafe {
            match tag {
                StrTag::Shared => self.drop_as::<Shared>(),
                StrTag::Boxed => self.drop_as::<Boxed>(),
                StrTag::Concat => self.drop_as::<Concat>(),
                StrTag::Inline | StrTag::Literal | StrTag::Empty => {}
            }
        };
    }
}

// Why UnsafeCell? We want something that wont increase the size of StrRep, but we also need to
// mutate it in-place. We can *almost* just use Cell here, but we cannot implement Clone behind
// cell.
#[derive(Default)]
#[repr(transparent)]
pub struct Str<'a>(UnsafeCell<StrRep<'a>>);

impl<'a> Str<'a> {
    unsafe fn rep(&self) -> &StrRep<'a> {
        &*self.0.get()
    }
    unsafe fn rep_mut(&self) -> &mut StrRep<'a> {
        &mut *self.0.get()
    }
    // We rely on string literals having trivial drops for LLVM codegen, as they may be dropped
    // repeatedly.
    pub fn drop_is_trivial(&self) -> bool {
        match unsafe { self.rep() }.get_tag() {
            StrTag::Empty | StrTag::Literal | StrTag::Inline => true,
            StrTag::Shared | StrTag::Concat | StrTag::Boxed => false,
        }
    }

    // leaks `self` unless you transmute it back. This is used in LLVM codegen
    pub fn into_bits(self) -> u128 {
        unsafe { mem::transmute::<Str<'a>, u128>(self) }
    }

    pub fn split(&self, pat: &Regex, mut push: impl FnMut(Str<'a>)) {
        self.with_str(|s| {
            if s.len() == 0 {
                return;
            }
            let mut prev = 0;
            for m in pat.find_iter(s) {
                // Awk will trim whitespace off the beginning of a line and not create an empty
                // field in $1, but this doesn't happen for other patterns: leading ','s when FS=,
                // do create empty fields, for example.
                if m.start() == 0 && s.chars().next().unwrap().is_whitespace() {
                    prev = m.end();
                    continue;
                }
                push(self.slice(prev, m.start()));
                prev = m.end();
            }
            push(self.slice(prev, s.len()));
        });
    }

    pub fn join<'b>(&self, mut ss: impl Iterator<Item = &'b Str<'a>>) -> Str<'a>
    where
        'a: 'b,
    {
        let mut res = if let Some(s) = ss.next() {
            s.clone()
        } else {
            return Default::default();
        };
        for s in ss {
            res = Str::concat(res.clone(), Str::concat(self.clone(), s.clone()));
        }
        res
    }

    pub fn subst_first(&self, pat: &Regex, subst: &Str<'a>) -> (Str<'a>, bool) {
        self.with_str(|s| {
            subst.with_str(|subst| {
                if let Some(m) = pat.find(s) {
                    let mut buf = DynamicBuf::new(s.len());
                    buf.write(&s.as_bytes()[0..m.start()]).unwrap();
                    buf.write(subst.as_bytes()).unwrap();
                    buf.write(&s.as_bytes()[m.end()..s.len()]).unwrap();
                    (unsafe { buf.into_str() }, true)
                } else {
                    (self.clone(), false)
                }
            })
        })
    }

    pub fn subst_all(&self, pat: &Regex, subst: &Str<'a>) -> (Str<'a>, Int) {
        self.with_str(|s| {
            subst.with_str(|subst| {
                let mut buf = DynamicBuf::new(0);
                let mut prev = 0;
                let mut count = 0;
                for m in pat.find_iter(s) {
                    buf.write(&s.as_bytes()[prev..m.start()]).unwrap();
                    buf.write(subst.as_bytes()).unwrap();
                    prev = m.end();
                    count += 1;
                }
                if count == 0 {
                    (self.clone(), count)
                } else {
                    buf.write(&s.as_bytes()[prev..s.len()]).unwrap();
                    (unsafe { buf.into_str() }, count)
                }
            })
        })
    }

    pub fn len(&self) -> usize {
        unsafe { self.rep_mut() }.len()
    }

    pub fn concat(left: Str<'a>, right: Str<'a>) -> Str<'a> {
        let llen = left.len();
        if llen == 0 {
            return right;
        }
        let rlen = right.len();
        if rlen == 0 {
            return left;
        }
        let new_len = llen + rlen;
        if new_len <= MAX_INLINE_SIZE {
            let mut b = DynamicBuf::new(0);
            unsafe {
                b.write(&*left.get_bytes()).unwrap();
                b.write(&*right.get_bytes()).unwrap();
                b.into_str()
            }
        } else {
            // TODO: we can add another case here. If `left` is boxed and has a refcount of 1, we
            // can move it into a dynamicbuf and push `right` onto it, avoiding the heap
            // allocation. We _only_ want to do this if we reevaluate the `realloc` that DynamicBuf
            // does when you convert it back into a string, though. We would have to keep a
            // capacity around as well as a length.
            let concat = Concat {
                len: new_len as u64,
                inner: Rc::new(ConcatInner { left, right }),
            };
            Str::from_rep(concat.into())
        }
    }

    fn from_rep(rep: StrRep<'a>) -> Str<'a> {
        Str(UnsafeCell::new(rep))
    }

    // This helper method assumes:
    // * that from and to cannot overflow when moved to u32s/shared/etc.
    // * that any CONCATs have been forced away.
    // * to - from > MAX_INLINE_SIZE
    unsafe fn slice_nooverflow(&self, from: usize, to: usize) -> Str<'a> {
        let rep = self.rep_mut();
        let tag = rep.get_tag();
        let new_rep = match tag {
            StrTag::Shared => rep.view_as(|s: &Shared| {
                let start = s.start + from as u32;
                let end = s.start + to as u32;
                Shared {
                    start,
                    end,
                    buf: s.buf.clone(),
                }
                .into()
            }),
            StrTag::Boxed => rep.view_as(|b: &Boxed| {
                Shared {
                    start: from as u32,
                    end: to as u32,
                    buf: b.buf.clone(),
                }
                .into()
            }),
            StrTag::Literal => rep.view_as(|l: &Literal| {
                let new_ptr = l.ptr.offset(from as isize);
                let new_len = (to - from) as u64;
                Literal {
                    len: new_len,
                    ptr: new_ptr,
                    _marker: PhantomData,
                }
                .into()
            }),
            StrTag::Empty | StrTag::Inline | StrTag::Concat => unreachable!(),
        };
        Str::from_rep(new_rep)
    }

    unsafe fn slice_internal(&self, from: usize, to: usize) -> Str<'a> {
        assert!(from <= to);
        if from == to {
            return Default::default();
        }
        let len = self.len();
        assert!(
            to <= len,
            "invalid args to slice: range [{},{}) with len {}",
            from,
            to,
            len
        );
        let new_len = to - from;
        if new_len <= MAX_INLINE_SIZE {
            return Str::from_rep(Inline::from_unchecked(&(*self.get_bytes())[from..to]).into());
        }
        let tag = self.rep().get_tag();
        let u32_max = u32::max_value() as usize;
        let mut may_overflow = to > u32_max || from > u32_max;
        if !may_overflow && tag == StrTag::Shared {
            // If we are taking a slice of an existing slice, then we can overflow by adding the
            // starts and ends together.
            may_overflow = self.rep_mut().view_as(|s: &Shared| {
                (s.start as usize + from) > u32_max || (s.start as usize + to) > u32_max
            });
        }
        // Slices of literals are addressed with 64 bits.
        may_overflow = may_overflow && tag != StrTag::Literal;
        if may_overflow {
            // uncommon case: we cannot represent a Shared value. We need to copy and box the value
            // instead.
            // TODO: We can optimize cases when we are getting suffixes of Literal values
            // by creating new ones with offset pointers. This doesn't seem worth optimizing right
            // now, but we may want to in the future.
            self.force();
            let rep = self.rep_mut();
            let tag = rep.get_tag();
            // All other variants ruled out by how large `self` is and the fact that we
            // just called `force`
            debug_assert_eq!(tag, StrTag::Boxed);
            return Str::from_rep(rep.view_as(|b: &Boxed| {
                let buf = Buf::read_from_raw(b.buf.as_ptr().offset(from as isize), new_len);
                Boxed {
                    len: new_len as u64,
                    buf,
                }
                .into()
            }));
        }

        // Force concat up here so we don't have to worry about aliasing `rep` in slice_nooverflow.
        if let StrTag::Concat = tag {
            self.force()
        }
        self.slice_nooverflow(from, to)
    }

    pub fn slice(&self, from: usize, to: usize) -> Str<'a> {
        // TODO: consider returning a result here so we can error out in a more graceful way.
        {
            use super::utf8::is_char_boundary;
            let bs = unsafe { &*self.get_bytes() };
            assert!(
                (from == to && to == bs.len()) || from < bs.len(),
                "internal error: invalid index len={}, from={}, to={}",
                bs.len(),
                from,
                to,
            );
            assert!(to <= bs.len(), "internal error: invalid index");
            assert!(
                (from == to) || is_char_boundary(bs[from]),
                "must take substring at char boundary"
            );
            assert!(
                to == bs.len() || is_char_boundary(bs[to]),
                "must take substring at char boundary"
            );
        }
        unsafe { self.slice_internal(from, to) }
    }

    // Why is [with_str] safe and [force] unsafe? Let's go case-by-case for the state of `self`
    // EMPTY:  no data is passed into `f`.
    // BOXED:  The function signature ensures that no string references can "escape" `f`, and `self`
    //         will persist for the function body, which will keep the underlying buffer alive.
    // CONCAT: We `force` these strings, so they will be BOXED.
    // SHARED: This one is tricky. It may seem to be covered by the BOXED case, but the difference
    //         is that shared strings give up there references to the underlying buffer if they get
    //         forced. So if we did s.with_str(|x| { /* force s */; *x}), then *x is a
    //         use-after-free!
    //
    //         This is why [force] is unsafe. As written, no safe method will force a SHARED Str.
    //         If we add force to a public API (e.g. for garbage collection), we'll need to ensure
    //         that we don't call with_str around it, or clone the string before forcing.

    unsafe fn force(&self) {
        let (tag, len) = {
            let rep = self.rep_mut();
            (rep.get_tag(), rep.len())
        };
        if tag.forced() {
            return;
        }
        let mut whead = 0;
        let mut res = UniqueBuf::new(len);
        macro_rules! push_bytes {
            ($slice:expr, [$from:expr, $to:expr]) => {{
                let from = $from;
                let slen = $to - from;
                push_bytes!(&$slice[$from], slen);
            }};
            ($at:expr, $len:expr) => {{
                let slen = $len;
                debug_assert!((len - whead) >= slen);
                let head = &mut res.as_mut_bytes()[whead];
                ptr::copy_nonoverlapping($at, head, slen);
                whead += slen;
            }};
        }
        let mut todos = SmallVec::<[Str<'a>; 16]>::new();
        let mut cur: Str<'a> = self.clone();
        let new_rep: StrRep<'a> = 'outer: loop {
            let rep = cur.rep_mut();
            let tag = rep.get_tag();
            cur = loop {
                match tag {
                    StrTag::Empty => {}
                    StrTag::Inline => rep.view_as(|i: &Inline| {
                        push_bytes!(i.bytes(), [0, i.len()]);
                    }),
                    StrTag::Literal => rep.view_as(|l: &Literal| {
                        push_bytes!(l.ptr, l.len as usize);
                    }),
                    StrTag::Boxed => rep.view_as(|b: &Boxed| {
                        push_bytes!(b.buf.as_bytes(), [0, b.len as usize]);
                    }),
                    StrTag::Shared => rep.view_as(|s: &Shared| {
                        push_bytes!(s.buf.as_bytes(), [s.start as usize, s.end as usize]);
                    }),
                    StrTag::Concat => {
                        break rep.view_as(|c: &Concat| {
                            todos.push(c.inner.right.clone());
                            c.inner.left.clone()
                        })
                    }
                }
                if let Some(c) = todos.pop() {
                    break c;
                }
                break 'outer Boxed {
                    len: len as u64,
                    buf: res.into_buf(),
                }
                .into();
            };
        };
        *self.rep_mut() = new_rep;
    }

    // Avoid using this function; subsequent immutable calls to &self can invalidate the pointer.
    pub fn get_bytes(&self) -> *const [u8] {
        let rep = unsafe { self.rep_mut() };
        let tag = rep.get_tag();
        unsafe {
            match tag {
                StrTag::Empty => &[],
                StrTag::Inline => rep.view_as(|i: &Inline| i.bytes() as *const _),
                StrTag::Literal => rep.view_as(|lit: &Literal| {
                    slice::from_raw_parts(lit.ptr, lit.len as usize) as *const _
                }),
                StrTag::Shared => rep.view_as(|s: &Shared| {
                    &s.buf.as_bytes()[s.start as usize..s.end as usize] as *const _
                }),
                StrTag::Boxed => rep.view_as(|b: &Boxed| b.buf.as_bytes() as *const _),
                StrTag::Concat => {
                    self.force();
                    self.get_bytes()
                }
            }
        }
    }
    pub fn get_raw_str(&self) -> *const str {
        unsafe { str::from_utf8_unchecked(&*self.get_bytes()) }
    }

    pub fn with_str<R>(&self, f: impl FnOnce(&str) -> R) -> R {
        let raw = self.get_raw_str();
        unsafe { f(&*raw) }
    }
    pub fn unmoor(self) -> Str<'static> {
        let rep = unsafe { self.rep_mut() };
        let tag = rep.get_tag();
        if let StrTag::Literal = tag {
            let new_rep = unsafe {
                rep.view_as(|lit: &Literal| {
                    let buf = Buf::read_from_raw(lit.ptr, lit.len as usize);
                    Boxed { len: lit.len, buf }.into()
                })
            };
            *rep = new_rep;
        }
        unsafe { mem::transmute::<Str<'a>, Str<'static>>(self) }
    }
}

impl<'a> Clone for Str<'a> {
    fn clone(&self) -> Str<'a> {
        let rep = unsafe { self.rep_mut() };
        let tag = rep.get_tag();
        let cloned_rep: StrRep<'a> = unsafe {
            match tag {
                StrTag::Empty | StrTag::Literal | StrTag::Inline => rep.copy(),
                StrTag::Shared => rep.view_as(|s: &Shared| s.clone()).into(),
                StrTag::Boxed => rep.view_as(|b: &Boxed| b.clone()).into(),
                StrTag::Concat => rep.view_as(|c: &Concat<'a>| c.clone()).into(),
            }
        };
        Str(UnsafeCell::new(cloned_rep))
    }
}

impl<'a> PartialEq for Str<'a> {
    fn eq(&self, other: &Str<'a>) -> bool {
        // If the bits are the same, then the strings are equal.
        if unsafe { self.rep() == other.rep() } {
            return true;
        }
        // TODO: we could intern these strings if they wind up equal.
        self.with_str(|s1| other.with_str(|s2| s1 == s2))
    }
}

impl<'a> Eq for Str<'a> {}

impl<'a> Hash for Str<'a> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.with_str(|s| s.hash(state))
    }
}

impl<'a> From<&'a str> for Str<'a> {
    fn from(s: &'a str) -> Str<'a> {
        if s.len() == 0 {
            Default::default()
        } else if s.len() <= MAX_INLINE_SIZE {
            Str::from_rep(unsafe { Inline::from_raw(s.as_ptr(), s.len()).into() })
        } else if s.as_ptr() as usize & 0x7 != 0 {
            // Strings are not guaranteed to be word aligned. Copy over strings that aren't. This
            // is more important for tests; most of the places that literals can come from in an
            // awk program will hand out aligned pointers.
            let buf = Buf::read_from_str(s);
            let boxed = Boxed {
                len: s.len() as u64,
                buf,
            };
            Str::from_rep(boxed.into())
        } else {
            let literal = Literal {
                len: s.len() as u64,
                ptr: s.as_ptr(),
                _marker: PhantomData,
            };
            Str::from_rep(literal.into())
        }
    }
}

impl<'a> From<String> for Str<'a> {
    fn from(s: String) -> Str<'a> {
        if s.len() == 0 {
            return Default::default();
        }
        let buf = Buf::read_from_str(s.as_str());
        let boxed = Boxed {
            len: s.len() as u64,
            buf,
        };
        Str::from_rep(boxed.into())
    }
}

// For numbers, we are careful to check if a number only requires 15 digits or fewer to be
// represented. This allows us to trigger the "Inline" variant and avoid a heap allocation,
// sometimes at the expenseof a small copy.

impl<'a> From<Int> for Str<'a> {
    fn from(i: Int) -> Str<'a> {
        let digit_guess = if i >= 1000000000000000 || i <= -100000000000000 {
            // Allocate on the heap; this is the maximum length we expect to see.
            21
        } else {
            // We'll allocate this inline.
            0
        };
        let mut b = DynamicBuf::new(digit_guess);
        write!(&mut b, "{}", i).unwrap();
        unsafe { b.into_str() }
    }
}

impl<'a> From<Float> for Str<'a> {
    fn from(f: Float) -> Str<'a> {
        // Per ryu's documentation, we will only ever use 24 bytes when printing an f64.
        if f.is_finite() {
            // TODO: fix this. Read it into a stack buffer, then write it in.
            let mut ryubuf = ryu::Buffer::new();
            let s = ryubuf.format(f);
            let mut b = DynamicBuf::new(s.len());
            b.write(s.as_bytes()).unwrap();
            unsafe { b.into_str() }
        } else {
            let mut b = DynamicBuf::new(0);
            write!(&mut b, "{}", f).unwrap();
            unsafe { b.into_str() }
        }
    }
}

impl Str<'static> {
    // Why have this? Parts of the runtime hold onto a Str<'static> to avoid adding a lifetime
    // parameter to the struct.
    pub fn upcast<'a>(self) -> Str<'a> {
        unsafe { mem::transmute::<Str<'static>, Str<'a>>(self) }
    }
    pub fn upcast_ref<'a>(&self) -> &Str<'a> {
        unsafe { mem::transmute::<&Str<'static>, &Str<'a>>(self) }
    }
}

#[repr(C)]
struct BufHeader {
    size: usize,
    // We only have "strong counts"
    count: Cell<usize>,
}

#[repr(transparent)]
pub struct UniqueBuf(*mut BufHeader);

pub struct DynamicBufHeap {
    data: UniqueBuf,
    write_head: usize,
}

impl DynamicBufHeap {
    pub fn new(size: usize) -> DynamicBufHeap {
        DynamicBufHeap {
            data: UniqueBuf::new(size),
            write_head: 0,
        }
    }
    fn size(&self) -> usize {
        unsafe { (*self.data.0).size }
    }
    pub unsafe fn into_str<'a>(mut self) -> Str<'a> {
        // Shrink the buffer to fit.
        self.realloc(self.write_head);
        self.data.into_buf().into_str()
    }
    unsafe fn realloc(&mut self, new_cap: usize) {
        let cap = self.size();
        let new_buf = realloc(
            self.data.0 as *mut u8,
            UniqueBuf::layout(cap),
            UniqueBuf::layout(new_cap).size(),
        ) as *mut BufHeader;
        (*new_buf).size = new_cap;
        self.data.0 = new_buf;
    }
}

impl Write for DynamicBufHeap {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let cap = self.size();
        let remaining = cap - self.write_head;
        unsafe {
            if remaining < buf.len() {
                let new_cap = std::cmp::max(buf.len(), cap * 2);
                self.realloc(new_cap);
                ptr::copy(
                    buf.as_ptr(),
                    self.data.as_mut_ptr().offset(self.write_head as isize),
                    buf.len(),
                );
            // NB: even after copying, there may be uninitialized memory at the tail of the
            // buffer. We enforce that this memory is never read by doing a realloc(write_head)
            // before moving this into a Buf. Before then, we don't read the underlying data at
            // all.
            } else {
                ptr::copy(
                    buf.as_ptr(),
                    self.data.as_mut_ptr().offset(self.write_head as isize),
                    buf.len(),
                )
            }
        };
        self.write_head += buf.len();
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub enum DynamicBuf {
    Inline(smallvec::SmallVec<[u8; MAX_INLINE_SIZE]>),
    Heap(DynamicBufHeap),
}

impl DynamicBuf {
    pub fn new(size: usize) -> DynamicBuf {
        if size <= MAX_INLINE_SIZE {
            DynamicBuf::Inline(Default::default())
        } else {
            DynamicBuf::Heap(DynamicBufHeap::new(size))
        }
    }
    pub unsafe fn into_str<'a>(self) -> Str<'a> {
        match self {
            DynamicBuf::Inline(sv) => Str::from_rep(Inline::from_unchecked(&sv[..]).into()),
            DynamicBuf::Heap(dbuf) => dbuf.into_str(),
        }
    }
}

impl Write for DynamicBuf {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            DynamicBuf::Inline(sv) => {
                let new_len = sv.len() + buf.len();
                if sv.len() + buf.len() > MAX_INLINE_SIZE {
                    let mut heap = DynamicBufHeap::new(new_len);
                    heap.write(&sv[..]).unwrap();
                    heap.write(buf).unwrap();
                    *self = DynamicBuf::Heap(heap);
                } else {
                    sv.extend(buf.iter().cloned());
                }
                Ok(buf.len())
            }
            DynamicBuf::Heap(dbuf) => dbuf.write(buf),
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[repr(transparent)]
pub struct Buf(*const BufHeader);

impl Clone for Buf {
    fn clone(&self) -> Buf {
        let header: &BufHeader = unsafe { &(*self.0) };
        let cur = header.count.get();
        header.count.set(cur + 1);
        Buf(self.0)
    }
}

impl Drop for UniqueBuf {
    fn drop(&mut self) {
        let header: &mut BufHeader = unsafe { &mut (*self.0) };
        debug_assert_eq!(header.count.get(), 1);
        unsafe { dealloc(self.0 as *mut u8, UniqueBuf::layout(header.size)) }
    }
}

impl Drop for Buf {
    fn drop(&mut self) {
        let header: &BufHeader = unsafe { &(*self.0) };
        let cur = header.count.get();
        debug_assert!(cur > 0);
        if cur == 1 {
            mem::drop(UniqueBuf(self.0 as *mut _));
            return;
        }
        header.count.set(cur - 1);
    }
}

impl UniqueBuf {
    fn layout(size: usize) -> Layout {
        Layout::from_size_align(
            size + mem::size_of::<BufHeader>(),
            mem::align_of::<BufHeader>(),
        )
        .unwrap()
    }
    pub fn new(size: usize) -> UniqueBuf {
        let layout = UniqueBuf::layout(size);
        unsafe {
            let alloced = alloc_zeroed(layout) as *mut BufHeader;
            assert!(!alloced.is_null());
            ptr::write(
                alloced,
                BufHeader {
                    size,
                    count: Cell::new(1),
                },
            );
            UniqueBuf(alloced)
        }
    }
    pub fn as_mut_bytes(&mut self) -> &mut [u8] {
        let header: &BufHeader = unsafe { &(*self.0) };
        debug_assert_eq!(header.count.get(), 1);
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), header.size) }
    }
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        let header: &BufHeader = unsafe { &(*self.0) };
        debug_assert_eq!(header.count.get(), 1);
        unsafe { self.0.offset(1) as *mut u8 }
    }
    pub fn into_buf(self) -> Buf {
        let res = Buf(self.0);
        mem::forget(self);
        res
    }
}

impl Buf {
    // Unsafe because it assumes valid utf8.
    pub unsafe fn into_str<'a>(self) -> Str<'a> {
        Str::from_rep(
            Boxed {
                len: self.len() as u64,
                buf: self,
            }
            .into(),
        )
    }
    pub fn len(&self) -> usize {
        unsafe { &(*self.0) }.size
    }
    pub fn as_bytes(&self) -> &[u8] {
        let size = self.len();
        unsafe { slice::from_raw_parts(self.as_ptr(), size) }
    }

    pub fn as_ptr(&self) -> *const u8 {
        unsafe { self.0.offset(1) as *const u8 }
    }

    pub unsafe fn read_from_raw(ptr: *const u8, len: usize) -> Buf {
        let mut ubuf = UniqueBuf::new(len);
        ptr::copy_nonoverlapping(ptr, ubuf.as_mut_ptr(), len);
        ubuf.into_buf()
    }

    pub fn read_from_str(s: &str) -> Buf {
        let res = unsafe { Buf::read_from_raw(s.as_ptr(), s.len()) };
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inline_basics() {
        let test_str = "hello there";
        unsafe {
            let i = Inline::from_unchecked(test_str.as_bytes());
            assert_eq!(test_str, str::from_utf8(i.bytes()).unwrap());
        }

        let s: Str = "hi there".into();
        assert_eq!(unsafe { s.rep().get_tag() }, StrTag::Inline);
        let s1 = s.slice(0, 1);
        assert_eq!(unsafe { s1.rep().get_tag() }, StrTag::Inline);
        s1.with_str(|s1| assert_eq!(s1, "h"));
    }

    #[test]
    fn basic_behavior() {
        let base_1 = "hi there fellow";
        let base_2 = "how are you?";
        let base_3 = "hi there fellowhow are you?";
        let s1 = Str::from(base_1);
        let s2 = Str::from(base_2);
        let s3 = Str::from(base_3);
        s1.with_str(|s| assert_eq!(s, base_1));
        s2.with_str(|s| assert_eq!(s, base_2, "{:?}", s2));
        s3.with_str(|s| assert_eq!(s, base_3));

        let s4 = Str::concat(s1, s2.clone());
        assert_eq!(s3, s4);
        s4.with_str(|s| assert_eq!(s, base_3));
        let s5 = Str::concat(
            Str::concat(Str::from("hi"), Str::from(" there")),
            Str::concat(
                Str::from(" "),
                Str::concat(Str::from("fel"), Str::from("low")),
            ),
        );
        s5.with_str(|s| assert_eq!(s, base_1));

        // Do this multiple times to play with the refcount.
        assert_eq!(s2.slice(1, 4), s3.slice(16, 19));
        assert_eq!(s2.slice(2, 6), s3.slice(17, 21));
    }

    fn test_str_split(pat: &Regex, base: &str) {
        let s = Str::from(base);
        let want = pat
            .split(base)
            .skip_while(|x| x.len() == 0)
            .collect::<Vec<_>>();
        let mut got = Vec::new();
        s.split(&pat, |sub| got.push(sub));
        let total_got = got.len();
        let total = want.len();
        for (g, w) in got.iter().cloned().zip(want.iter().cloned()) {
            assert_eq!(g, Str::from(w));
        }
        if total_got > total {
            // We want there to be trailing empty fields in this case.
            for s in &got[total..] {
                assert_eq!(s.len(), 0);
            }
        } else {
            assert_eq!(total_got, total, "got={:?} vs want={:?}", got, want);
        }
    }

    #[test]
    fn basic_splitting() {
        let pat0 = Regex::new(",").unwrap();
        test_str_split(&pat0, "what,is,,,up,");
        let pat = Regex::new(r#"[ \t]"#).unwrap();
        test_str_split(&pat, "what is \t up ");
    }

    #[test]
    fn split_long_string() {
        let pat = Regex::new(r#"[ \t]"#).unwrap();
        test_str_split(&pat, crate::test_string_constants::PRIDE_PREJUDICE_CH2);
    }

    #[test]
    fn dynamic_string() {
        let mut d = DynamicBuf::new(0);
        write!(
            &mut d,
            "This is the first part of the string {}\n",
            "with formatting and everything!"
        )
        .unwrap();
        write!(&mut d, "And this is the second part").unwrap();
        let s = unsafe { d.into_str() };
        s.with_str(|s| {
            assert_eq!(
                s,
                r#"This is the first part of the string with formatting and everything!
And this is the second part"#
            )
        });
    }

    #[test]
    fn subst() {
        let s1: Str = "String number one".into();
        let s2: Str = "m".into();
        let re1 = Regex::new("n").unwrap();
        let (s3, n1) = s1.subst_all(&re1, &s2);
        assert_eq!(n1, 3);
        s3.with_str(|s| assert_eq!(s, "Strimg mumber ome"));

        let re2 = Regex::new("xxyz").unwrap();
        let (s4, n2) = s3.subst_all(&re2, &s2);
        assert_eq!(n2, 0);
        assert_eq!(s3, s4);

        let empty = Str::default();
        let (s5, n3) = empty.subst_all(&re1, &s2);
        assert_eq!(n3, 0);
        assert_eq!(empty, s5);

        let s6: Str = "xxyz substituted into another xxyz".into();
        let (s7, subbed) = s6.subst_first(&re2, &s1);
        s7.with_str(|s| assert_eq!(s, "String number one substituted into another xxyz"));
        assert!(subbed);
    }
}

mod formatting {
    use super::*;
    use std::fmt::{self, Debug, Display, Formatter};

    impl<'a> Display for Str<'a> {
        fn fmt(&self, f: &mut Formatter) -> fmt::Result {
            self.with_str(|s| write!(f, "{}", s))
        }
    }

    impl<'a> Debug for Str<'a> {
        fn fmt(&self, f: &mut Formatter) -> fmt::Result {
            unsafe {
                let rep = self.rep_mut();
                match rep.get_tag() {
                    StrTag::Empty => write!(f, "Str(EMPTY)"),
                    StrTag::Inline => {
                        rep.view_as(|i: &Inline| write!(f, "Str(Inline({:?}))", i.bytes()))
                    }
                    StrTag::Literal => rep.view_as(|l: &Literal| write!(f, "Str({:?})", l)),
                    StrTag::Shared => rep.view_as(|s: &Shared| write!(f, "Str({:?})", s)),
                    StrTag::Concat => rep.view_as(|c: &Concat| write!(f, "Str({:?})", c)),
                    StrTag::Boxed => rep.view_as(|b: &Boxed| write!(f, "Str({:?})", b)),
                }?
            }
            write!(f, "/[disp=<{}>]", self)
        }
    }

    impl<'a> Debug for Literal<'a> {
        fn fmt(&self, f: &mut Formatter) -> fmt::Result {
            write!(
                f,
                "Literal {{ len: {}, ptr: {:x}=>{:?} }}",
                self.len,
                self.ptr as usize,
                str::from_utf8(unsafe { slice::from_raw_parts(self.ptr, self.len as usize) })
                    .unwrap(),
            )
        }
    }

    impl<'a> Debug for Buf {
        fn fmt(&self, f: &mut Formatter) -> fmt::Result {
            let header = unsafe { &*self.0 };
            write!(
                f,
                "Buf {{ size: {}, count: {}, contents: {:?} }}",
                header.size,
                header.count.get(),
                str::from_utf8(self.as_bytes()).unwrap(),
            )
        }
    }
}
