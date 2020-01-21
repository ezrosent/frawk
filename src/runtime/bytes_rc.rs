use crate::smallvec::SmallVec;

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::cell::{Cell, UnsafeCell};
use std::marker::PhantomData;
use std::mem;
use std::ptr;
use std::rc::Rc;
use std::slice;
use std::str;

const EMPTY: usize = 0;
const SHARED: usize = 1;
const LITERAL: usize = 2;
const CONCAT: usize = 3;
const BOXED: usize = 4;
const NUM_VARIANTS: usize = 5;

// Why the repr(C)? We may rely on the lengths coming first.

#[derive(Clone)]
#[repr(C)]
struct Literal<'a> {
    len: u64,
    ptr: *const u8,
    _marker: PhantomData<&'a ()>,
}

#[derive(Clone, Debug)]
#[repr(C)]
struct Boxed {
    len: u64,
    buf: Buf,
}

#[derive(Clone, Debug)]
#[repr(C)]
struct Shared {
    start: u32,
    end: u32,
    buf: Buf,
}

#[derive(Clone, Debug)]
struct ConcatInner<'a> {
    left: Str<'a>,
    right: Str<'a>,
}
#[derive(Clone, Debug)]
#[repr(C)]
struct Concat<'a> {
    len: u64,
    inner: Rc<ConcatInner<'a>>,
}

#[derive(Default, PartialEq, Eq)]
#[repr(C)]
struct StrRep<'a> {
    low: u64,
    hi: usize,
    _marker: PhantomData<&'a ()>,
}

impl<'a> PartialEq for Str<'a> {
    fn eq(&self, other: &Str<'a>) -> bool {
        if unsafe { &*self.0.get() == &*other.0.get() } {
            return true;
        }
        self.with_str(|s1| other.with_str(|s2| s1 == s2))
    }
}

impl<'a> StrRep<'a> {
    fn get_tag(&self) -> usize {
        let tag = self.hi & 0x7;
        debug_assert!(tag < NUM_VARIANTS);
        tag
    }
}

macro_rules! impl_tagged_from {
    ($from:ty, $tag:expr) => {
        impl<'a> From<$from> for StrRep<'a> {
            fn from(s: $from) -> StrRep<'a> {
                let mut rep = unsafe { mem::transmute::<$from, StrRep>(s) };
                rep.hi |= $tag;
                rep
            }
        }
    };
}

impl_tagged_from!(Shared, SHARED);
impl_tagged_from!(Concat<'a>, CONCAT);
impl_tagged_from!(Literal<'a>, LITERAL);
impl_tagged_from!(Boxed, BOXED);

impl<'a> StrRep<'a> {
    fn len(&mut self) -> usize {
        match self.get_tag() {
            EMPTY => 0,
            BOXED | LITERAL | CONCAT => self.low as usize,
            SHARED => unsafe { self.view_as(|s: &Shared| s.end as usize - s.start as usize) },
            _ => unreachable!(),
        }
    }
    unsafe fn view_as<T, R>(&mut self, f: impl FnOnce(&T) -> R) -> R {
        let old = self.hi;
        self.hi = old & !0x7;
        let res = f(mem::transmute::<&mut StrRep<'a>, &T>(self));
        self.hi = old;
        res
    }
    unsafe fn view_as_mut<T, R>(&mut self, f: impl FnOnce(&mut T) -> R) -> R {
        let old = self.hi;
        self.hi = old & !0x7;
        let res = f(mem::transmute::<&mut StrRep<'a>, &mut T>(self));
        self.hi = old;
        res
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
                SHARED => self.view_as_mut(|s: &mut Shared| ptr::drop_in_place(s)),
                BOXED => self.view_as_mut(|b: &mut Boxed| ptr::drop_in_place(b)),
                CONCAT => self.view_as_mut(|c: &mut Concat<'a>| ptr::drop_in_place(c)),
                _ => {}
            }
        };
    }
}

impl<'a> From<&'a str> for Str<'a> {
    fn from(s: &'a str) -> Str<'a> {
        if s.len() == 0 {
            return Default::default();
        }
        if s.as_ptr() as usize & 0x7 != 0 {
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

// Why UnsafeCell? We want something that wont increase the size of StrRep, but we also need to
// mutate it in-place. We can *almost* just use Cell here, but we cannot implement Clone behind
// cell.
#[derive(Default)]
#[repr(transparent)]
pub struct Str<'a>(UnsafeCell<StrRep<'a>>);

impl<'a> Str<'a> {
    fn len(&self) -> usize {
        let rep = unsafe { &mut *self.0.get() };
        rep.len()
    }

    pub fn concat(left: Str<'a>, right: Str<'a>) -> Str<'a> {
        let concat = Concat {
            len: (left.len() + right.len()) as u64,
            inner: Rc::new(ConcatInner { left, right }),
        };
        Str::from_rep(concat.into())
    }

    fn from_rep(rep: StrRep<'a>) -> Str<'a> {
        Str(UnsafeCell::new(rep))
    }

    // This helper method assumes:
    // * that from and to cannot overflow when moved to u32s/shared/etc.
    // * that any CONCATs have been forced away.
    unsafe fn slice_nooverflow(&self, from: usize, to: usize) -> Str<'a> {
        let rep = &mut *self.0.get();
        let tag = rep.get_tag();
        let new_rep = match tag {
            EMPTY => return Default::default(),
            SHARED => rep.view_as(|s: &Shared| {
                let start = s.start + from as u32;
                let end = s.start + to as u32;
                Shared {
                    start,
                    end,
                    buf: s.buf.clone(),
                }
                .into()
            }),
            BOXED => rep.view_as(|b: &Boxed| {
                Shared {
                    start: from as u32,
                    end: to as u32,
                    buf: b.buf.clone(),
                }
                .into()
            }),
            LITERAL => rep.view_as(|l: &Literal| {
                Literal {
                    len: (to - from) as u64,
                    ptr: l.ptr.offset(from as isize),
                    _marker: PhantomData,
                }
                .into()
            }),
            _ => unreachable!(),
        };
        Str::from_rep(new_rep)
    }

    pub fn slice(&self, from: usize, to: usize) -> Str<'a> {
        assert!(from <= to);
        let len = self.len();
        assert!(to < len);
        let new_len = to - from;
        let tag = unsafe { &mut *self.0.get() }.get_tag();
        let u32_max = u32::max_value() as usize;
        let mut may_overflow = to > u32_max || from > u32_max;
        if !may_overflow && tag == SHARED {
            // If we are taking a slice of an existing slice, then we can overflow by adding the
            // starts and ends together.
            may_overflow = unsafe {
                (*self.0.get()).view_as(|s: &Shared| {
                    (s.start as usize + from) > u32_max || (s.start as usize + to) > u32_max
                })
            };
        }
        // Slices of literals are addressed with 64 bits.
        may_overflow = may_overflow && tag != LITERAL;
        if may_overflow {
            // uncommon case: we cannot represent a Shared value. We need to copy and box the value
            // instead.
            // TODO: We can optimize cases when we are getting suffixes of Literal values
            // by creating new ones with offset pointers. This doesn't seem worth optimizing right
            // now, but we may want to in the future.
            self.force();
            let rep = unsafe { &mut *self.0.get() };
            let tag = rep.get_tag();
            unsafe {
                // All other variants ruled out by how large `self` is and the fact that we
                // just called `force`
                debug_assert_eq!(tag, BOXED);
                return Str::from_rep(rep.view_as(|b: &Boxed| {
                    let buf = Buf::read_from_raw(b.buf.as_ptr().offset(from as isize), new_len);
                    Boxed {
                        len: new_len as u64,
                        buf,
                    }
                    .into()
                }));
            }
        }

        // Force concat up here so we don't have to worry about aliasing `rep` in slice_nooverflow.
        if let CONCAT = tag {
            self.force();
        }
        unsafe { self.slice_nooverflow(from, to) }
    }

    fn force(&self) {
        let (tag, len) = {
            let rep = unsafe { &mut *self.0.get() };
            (rep.get_tag(), rep.len())
        };
        if let LITERAL | BOXED | EMPTY = tag {
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
            unsafe {
                let rep = &mut (*cur.0.get());
                let tag = rep.get_tag();
                cur = loop {
                    match tag {
                        EMPTY => {}
                        LITERAL => rep.view_as(|l: &Literal| {
                            push_bytes!(l.ptr, l.len as usize);
                        }),
                        BOXED => rep.view_as(|b: &Boxed| {
                            push_bytes!(b.buf.as_bytes(), [0, b.len as usize]);
                        }),
                        SHARED => rep.view_as(|s: &Shared| {
                            push_bytes!(s.buf.as_bytes(), [s.start as usize, s.end as usize]);
                        }),
                        CONCAT => {
                            break rep.view_as(|c: &Concat| {
                                todos.push(c.inner.right.clone());
                                c.inner.left.clone()
                            })
                        }
                        _ => unreachable!(),
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
            }
        };
        *unsafe { &mut *self.0.get() } = new_rep;
    }
    pub fn with_str<R>(&self, mut f: impl FnMut(&str) -> R) -> R {
        let rep = unsafe { &mut *self.0.get() };
        let tag = rep.get_tag();
        unsafe {
            match tag {
                EMPTY => f(str::from_utf8_unchecked(&[])),
                LITERAL => rep.view_as(|lit: &Literal| {
                    f(str::from_utf8_unchecked(slice::from_raw_parts(
                        lit.ptr,
                        lit.len as usize,
                    )))
                }),
                SHARED => rep.view_as(|s: &Shared| {
                    f(str::from_utf8_unchecked(
                        &s.buf.as_bytes()[s.start as usize..s.end as usize],
                    ))
                }),
                BOXED => rep.view_as(|b: &Boxed| f(str::from_utf8_unchecked(b.buf.as_bytes()))),
                CONCAT => {
                    self.force();
                    return self.with_str(f);
                }
                _ => unreachable!(),
            }
        }
    }
}

impl<'a> Clone for Str<'a> {
    fn clone(&self) -> Str<'a> {
        let rep = unsafe { &mut *self.0.get() };
        let tag = rep.get_tag();
        let cloned_rep: StrRep<'a> = unsafe {
            match tag {
                EMPTY | LITERAL => rep.copy(),
                SHARED => rep.view_as(|s: &Shared| s.clone()).into(),
                BOXED => rep.view_as(|b: &Boxed| b.clone()).into(),
                CONCAT => rep.view_as(|c: &Concat<'a>| c.clone()).into(),
                _ => unreachable!(),
            }
        };
        Str(UnsafeCell::new(cloned_rep))
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
    pub fn as_bytes(&self) -> &[u8] {
        let size = unsafe { &(*self.0) }.size;
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

mod tests {
    use super::*;

    #[test]
    fn basic_behavior() {
        let base_1 = "hi there fellow";
        let base_2 = "how are you?";
        let base_3 = "hi there fellowhow are you?";
        let s1 = Str::from(base_1);
        let s2 = Str::from(base_2);
        let s3 = Str::from(base_3);
        s1.with_str(|s| assert_eq!(s, base_1));
        s2.with_str(|s| assert_eq!(s, base_2));
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
                let rep = &mut *self.0.get();
                match rep.get_tag() {
                    EMPTY => write!(f, "Str(EMPTY)"),
                    LITERAL => rep.view_as(|l: &Literal| write!(f, "Str({:?})", l)),
                    SHARED => rep.view_as(|s: &Shared| write!(f, "Str({:?})", s)),
                    CONCAT => rep.view_as(|c: &Concat| write!(f, "Str({:?})", c)),
                    BOXED => rep.view_as(|b: &Boxed| write!(f, "Str({:?})", b)),
                    _ => unreachable!(),
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
