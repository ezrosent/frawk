use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::mem;
use std::ptr;

#[derive(Clone)]
struct Shared {
    start: u32,
    end: u32,
    buf: Buf,
}

#[repr(C)]
struct BufHeader {
    size: usize,
    // We only use "strong counts"
    count: usize,
}

#[repr(transparent)]
pub struct UniqueBuf(*mut BufHeader);

#[repr(transparent)]
pub struct Buf(*mut BufHeader);

impl Clone for Buf {
    fn clone(&self) -> Buf {
        let header: &mut BufHeader = unsafe { &mut (*self.0) };
        header.count += 1;
        Buf(self.0)
    }
}

impl Drop for UniqueBuf {
    fn drop(&mut self) {
        let header: &mut BufHeader = unsafe { &mut (*self.0) };
        debug_assert_eq!(header.count, 1);
        unsafe { dealloc(self.0 as *mut u8, UniqueBuf::layout(header.size)) }
    }
}

impl Drop for Buf {
    fn drop(&mut self) {
        let header: &mut BufHeader = unsafe { &mut (*self.0) };
        if header.count == 1 {
            mem::drop(UniqueBuf(self.0));
            return;
        }
        header.count -= 1;
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
    pub fn new(size: usize) {
        let layout = UniqueBuf::layout(size);
        unsafe {
            let alloced = alloc_zeroed(layout) as *mut BufHeader;
            ptr::write(alloced, BufHeader { size, count: 1 });
            UniqueBuf(alloced)
        };
    }
    pub fn as_mut_bytes(&mut self) -> &mut [u8] {
        let header: &BufHeader = unsafe { &(*self.0) };
        debug_assert_eq!(header.count, 1);
        unsafe {
            let data_start = (self.0 as *mut u8).offset(mem::size_of::<BufHeader>() as isize);
            std::slice::from_raw_parts_mut(data_start, header.size)
        }
    }
    pub fn into_buf(self) -> Buf {
        Buf(self.0)
    }
}

impl Buf {
    pub fn as_bytes(&self) -> &[u8] {
        let size = unsafe { &(*self.0) }.size;
        unsafe {
            let data_start = (self.0 as *mut u8).offset(mem::size_of::<BufHeader>() as isize);
            std::slice::from_raw_parts_mut(data_start, size)
        }
    }
}

const EMPTY: usize = 0;
const SHARED: usize = 1;
const LITERAL: usize = 2;
const CONCAT: usize = 3;
// used only when a string is >=2^32 in size
const BOXED: usize = 4;
const NUM_VARIANTS: usize = 5;

/*
mod new {
    use std::cell::Cell;
    use std::marker::PhantomData;
    use std::mem;
    use std::rc::Rc;

    // for shared, we encode start and end as u32s, with the tagged pointer as the second u64.

    #[repr(transparent)]
    struct Literal<'a>(*const u8, PhantomData<&'a str>)

    struct ConcatInner<'a> {
        left: Str<'a>,
        right: Str<'a>,
    }

    const EMPTY: usize = 0;
    const SHARED: usize = 1;
    const LITERAL: usize = 2;
    const CONCAT: usize = 3;
    // used only when a string is >=2^32 in size
    const BOXED: usize = 4;
    const NUM_VARIANTS: usize = 5;

    #[repr(C)]
    struct Inner<'a> {
        // Usually length
        hi: usize,
        // Usually pointer
        lo: usize,
        marker: PhantomData<&'a ()>,
    };

    impl<'a> Default for Inner<'a> {
        fn default() -> Inner<'a> {
            Inner{hi: 0, lo: 0, marker: PhantomData, }
        }
    }

    // impl<'a> From<Rc<SharedStr>> for Inner<'a> {
    //     fn from(s: Rc<SharedStr>) -> Inner<'a> {
    //         unsafe {
    //             Inner(
    //                 mem::transmute::<Rc<SharedStr>, usize>(s) | SHARED,
    //                 PhantomData,
    //             )
    //         }
    //     }
    // }

    // impl<'a> From<Rc<ConcatInner<'a>>> for Inner<'a> {
    //     fn from(s: Rc<ConcatInner<'a>>) -> Inner<'a> {
    //         unsafe {
    //             Inner(
    //                 mem::transmute::<Rc<ConcatInner>, usize>(s) | CONCAT,
    //                 PhantomData,
    //             )
    //         }
    //     }
    // }
    // impl<'a> From<Rc<Literal<'a>>> for Inner<'a> {
    //     fn from(lit: Rc<Literal<'a>>) -> Inner<'a> {
    //         unsafe {
    //             Inner(
    //                 mem::transmute::<Rc<Literal<'a>>, usize>(lit) | LITERAL,
    //                 PhantomData,
    //             )
    //         }
    //     }
    // }

    impl<'a> From<String> for Inner<'a> {
        fn from(s: String) -> Inner<'a> {
            if s.len() == 0 {
                return Inner::default();
            }
            let rcd = Rc::new(s);
            let ptr = mem::transmute<Rc<String>, usize>(rcd) | SHARED;
            Rc::new(SharedStr {
                start: rcd.as_ptr(),
                len: rcd.len(),
                base: rcd.clone(),
            })
            .into()
        }
    }

    impl<'a> From<&'a str> for Inner<'a> {
        fn from(s: &'a str) -> Inner<'a> {
            Rc::new(Literal {
                ptr: s.as_ptr(),
                len: s.len(),
                marker: PhantomData,
            })
            .into()
        }
    }

    impl<'a> Clone for Inner<'a> {
        fn clone(&self) -> Inner<'a> {
            let tag = self.0 & 0x7;
            let addr = self.0 & !(0x7);
            debug_assert!(tag < NUM_VARIANTS);
            unsafe {
                match tag {
                    SHARED => mem::transmute::<usize, Rc<SharedStr>>(addr).clone().into(),
                    // for completeness. This drop should be trivial
                    LITERAL => mem::transmute::<usize, Rc<Literal<'a>>>(addr)
                        .clone()
                        .into(),
                    CONCAT => mem::transmute::<usize, Rc<ConcatInner<'a>>>(addr)
                        .clone()
                        .into(),
                    EMPTY => Inner::default(),
                    _ => unreachable!(),
                }
            }
        }
    }

    impl<'a> Drop for Inner<'a> {
        fn drop(&mut self) {
            let tag = self.0 & 0x7;
            let addr = self.0 & !(0x7);
            debug_assert!(tag < NUM_VARIANTS);
            unsafe {
                match tag {
                    // TODO fix
                    SHARED => mem::drop(mem::transmute::<usize, Rc<SharedStr>>(addr)),
                    // for completeness. This drop should be trivial
                    LITERAL => mem::drop(mem::transmute::<usize, Rc<Literal<'a>>>(addr)),
                    CONCAT => mem::drop(mem::transmute::<usize, Rc<ConcatInner<'a>>>(addr)),
                    _ => {}
                }
            }
        }
    }

    #[repr(transparent)]
    struct Str<'a>(Cell<Inner<'a>>);
} */
