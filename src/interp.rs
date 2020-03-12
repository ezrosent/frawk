use crate::bytecode::{Get, Instr, Label, Pop, Reg};

use crate::builtins::Variable;
use crate::common::{NumTy, Result};
use crate::compile::{self, Ty};
use crate::runtime::{self, Float, Int, LazyVec, Line, LineReader, Str};

use std::cmp;

type ClassicReader = runtime::splitter::Reader<Box<dyn std::io::Read>>;

#[derive(Default)]
pub(crate) struct Storage<T> {
    pub(crate) regs: Vec<T>,
    pub(crate) stack: Vec<T>,
}

pub(crate) struct Interp<'a, LR: LineReader = ClassicReader> {
    // index of `instrs` that contains "main"
    main_func: usize,
    instrs: Vec<Vec<Instr<'a>>>,
    stack: Vec<(usize /*function*/, Label /*instr*/)>,

    vars: runtime::Variables<'a>,

    line: LR::Line,
    split_line: LazyVec<Str<'a>>,

    regexes: runtime::RegexCache,
    write_files: runtime::FileWrite,
    read_files: runtime::FileRead<LR>,

    // TODO: should these be smallvec<[T; 32]>? We never add registers, so could we allocate one
    // contiguous region ahead of time?
    pub(crate) floats: Storage<Float>,
    pub(crate) ints: Storage<Int>,
    pub(crate) strs: Storage<Str<'a>>,
    pub(crate) maps_int_float: Storage<runtime::IntMap<Float>>,
    pub(crate) maps_int_int: Storage<runtime::IntMap<Int>>,
    pub(crate) maps_int_str: Storage<runtime::IntMap<Str<'a>>>,

    pub(crate) maps_str_float: Storage<runtime::StrMap<'a, Float>>,
    pub(crate) maps_str_int: Storage<runtime::StrMap<'a, Int>>,
    pub(crate) maps_str_str: Storage<runtime::StrMap<'a, Str<'a>>>,

    pub(crate) iters_int: Storage<runtime::Iter<Int>>,
    pub(crate) iters_str: Storage<runtime::Iter<Str<'a>>>,
}

fn default_of<T: Default>(n: usize) -> Storage<T> {
    let mut regs = Vec::new();
    regs.resize_with(n, Default::default);
    Storage {
        regs,
        stack: Default::default(),
    }
}

impl<'a> Interp<'a> {
    pub(crate) fn new(
        instrs: Vec<Vec<Instr<'a>>>,
        main_func: usize,
        regs: impl Fn(compile::Ty) -> usize,
        stdin: impl std::io::Read + 'static,
        stdout: impl std::io::Write + 'static,
    ) -> Interp<'a> {
        use compile::Ty::*;
        Interp {
            main_func,
            instrs,
            stack: Default::default(),
            floats: default_of(regs(Float)),
            ints: default_of(regs(Int)),
            strs: default_of(regs(Str)),
            vars: Default::default(),

            line: "".into(),
            split_line: LazyVec::new(),
            regexes: Default::default(),
            write_files: runtime::FileWrite::new(stdout),
            read_files: runtime::FileRead::new_transitional(stdin),

            maps_int_float: default_of(regs(MapIntFloat)),
            maps_int_int: default_of(regs(MapIntInt)),
            maps_int_str: default_of(regs(MapIntStr)),

            maps_str_float: default_of(regs(MapStrFloat)),
            maps_str_int: default_of(regs(MapStrInt)),
            maps_str_str: default_of(regs(MapStrStr)),

            iters_int: default_of(regs(IterInt)),
            iters_str: default_of(regs(IterStr)),
        }
    }
}

impl<'a, LR: LineReader> Interp<'a, LR> {
    pub(crate) fn instrs(&self) -> &Vec<Vec<Instr<'a>>> {
        &self.instrs
    }
    fn format_arg(&self, (reg, ty): (NumTy, Ty)) -> Result<runtime::FormatArg<'a>> {
        Ok(match ty {
            Ty::Str => self.get(Reg::<Str<'a>>::from(reg)).clone().into(),
            Ty::Int => self.get(Reg::<Int>::from(reg)).clone().into(),
            Ty::Float => self.get(Reg::<Float>::from(reg)).clone().into(),
            _ => return err!("non-scalar (s)printf argument type {:?}", ty),
        })
    }

    pub(crate) fn run(&mut self) -> Result<()> {
        use Instr::*;
        let newline: Str = "\n".into();
        let mut scratch: Vec<runtime::FormatArg> = Vec::new();
        // We are only accessing one vector at a time here, but it's hard to convince the borrow
        // checker of this fact, so we access the vectors through raw pointers.
        let mut cur_fn = self.main_func;
        let mut instrs = (&mut self.instrs[cur_fn]) as *mut Vec<Instr<'a>>;
        let mut cur = 0;
        'outer: loop {
            // must end with Halt
            cur = loop {
                debug_assert!(cur < unsafe { (*instrs).len() });
                use Variable::*;
                match unsafe { (*instrs).get_unchecked(cur) } {
                    StoreConstStr(sr, s) => {
                        let sr = *sr;
                        *self.get_mut(sr) = s.clone()
                    }
                    StoreConstInt(ir, i) => {
                        let ir = *ir;
                        *self.get_mut(ir) = *i
                    }
                    StoreConstFloat(fr, f) => {
                        let fr = *fr;
                        *self.get_mut(fr) = *f
                    }
                    IntToStr(sr, ir) => {
                        let s = runtime::convert::<_, Str>(*self.get(*ir));
                        let sr = *sr;
                        *self.get_mut(sr) = s;
                    }
                    FloatToStr(sr, fr) => {
                        let s = runtime::convert::<_, Str>(*self.get(*fr));
                        let sr = *sr;
                        *self.get_mut(sr) = s;
                    }
                    StrToInt(ir, sr) => {
                        let i = runtime::convert::<_, Int>(self.get(*sr));
                        let ir = *ir;
                        *self.get_mut(ir) = i;
                    }
                    StrToFloat(fr, sr) => {
                        let f = runtime::convert::<_, Float>(self.get(*sr));
                        let fr = *fr;
                        *self.get_mut(fr) = f;
                    }
                    FloatToInt(ir, fr) => {
                        let i = runtime::convert::<_, Int>(*self.get(*fr));
                        let ir = *ir;
                        *self.get_mut(ir) = i;
                    }
                    IntToFloat(fr, ir) => {
                        let f = runtime::convert::<_, Float>(*self.get(*ir));
                        let fr = *fr;
                        *self.get_mut(fr) = f;
                    }
                    AddInt(res, l, r) => {
                        let res = *res;
                        let l = *self.get(*l);
                        let r = *self.get(*r);
                        *self.get_mut(res) = l + r;
                    }
                    AddFloat(res, l, r) => {
                        let res = *res;
                        let l = *self.get(*l);
                        let r = *self.get(*r);
                        *self.get_mut(res) = l + r;
                    }
                    MulInt(res, l, r) => {
                        let res = *res;
                        let l = *self.get(*l);
                        let r = *self.get(*r);
                        *self.get_mut(res) = l * r;
                    }
                    MulFloat(res, l, r) => {
                        let res = *res;
                        let l = *self.get(*l);
                        let r = *self.get(*r);
                        *self.get_mut(res) = l * r;
                    }
                    MinusInt(res, l, r) => {
                        let res = *res;
                        let l = *self.get(*l);
                        let r = *self.get(*r);
                        *self.get_mut(res) = l - r;
                    }
                    MinusFloat(res, l, r) => {
                        let res = *res;
                        let l = *self.get(*l);
                        let r = *self.get(*r);
                        *self.get_mut(res) = l - r;
                    }
                    ModInt(res, l, r) => {
                        let res = *res;
                        let l = *self.get(*l);
                        let r = *self.get(*r);
                        *self.get_mut(res) = l % r;
                    }
                    ModFloat(res, l, r) => {
                        let res = *res;
                        let l = *self.get(*l);
                        let r = *self.get(*r);
                        *self.get_mut(res) = l % r;
                    }
                    Div(res, l, r) => {
                        let res = *res;
                        let l = *self.get(*l);
                        let r = *self.get(*r);
                        *self.get_mut(res) = l / r;
                    }
                    Not(res, ir) => {
                        let res = *res;
                        let i = *self.get(*ir);
                        *self.get_mut(res) = (i == 0) as Int;
                    }
                    NotStr(res, sr) => {
                        let res = *res;
                        let sr = *sr;
                        let is_empty = self.get(sr).with_str(|s| s.len() == 0);
                        *self.get_mut(res) = is_empty as Int;
                    }
                    NegInt(res, ir) => {
                        let res = *res;
                        let i = *self.get(*ir);
                        *self.get_mut(res) = -i;
                    }
                    NegFloat(res, fr) => {
                        let res = *res;
                        let f = *self.get(*fr);
                        *self.get_mut(res) = -f;
                    }
                    Concat(res, l, r) => {
                        let res = *res;
                        let l = self.get(*l).clone();
                        let r = self.get(*r).clone();
                        *self.get_mut(res) = Str::concat(l, r);
                    }
                    Match(res, l, r) => {
                        let res = *res;
                        let l = index(&self.strs, l);
                        let pat = index(&self.strs, r);
                        *self.get_mut(res) =
                            self.regexes.regex_match_loc(&mut self.vars, &pat, &l)? as Int;
                    }
                    IsMatch(res, l, r) => {
                        let res = *res;
                        let l = index(&self.strs, l);
                        let pat = index(&self.strs, r);
                        *self.get_mut(res) = self.regexes.is_regex_match(&pat, &l)? as Int;
                    }
                    LenStr(res, s) => {
                        let res = *res;
                        let s = *s;
                        // TODO consider doing a with_str here or enforce elsewhere that strings
                        // cannot exceed u32::max.
                        let len = self.get(s).len();
                        *self.get_mut(res) = len as Int;
                    }
                    Sub(res, pat, s, in_s) => {
                        let (subbed, new) = {
                            let pat = index(&self.strs, pat);
                            let s = index(&self.strs, s);
                            let in_s = index(&self.strs, in_s);
                            self.regexes.with_regex(pat, |re| in_s.subst_first(re, s))?
                        };
                        *index_mut(&mut self.strs, in_s) = subbed;
                        *index_mut(&mut self.ints, res) = new as Int;
                    }
                    GSub(res, pat, s, in_s) => {
                        let (subbed, subs_made) = {
                            let pat = index(&self.strs, pat);
                            let s = index(&self.strs, s);
                            let in_s = index(&self.strs, in_s);
                            self.regexes.with_regex(pat, |re| in_s.subst_all(re, s))?
                        };
                        *index_mut(&mut self.strs, in_s) = subbed;
                        *index_mut(&mut self.ints, res) = subs_made;
                    }
                    Substr(res, base, l, r) => {
                        let base = index(&self.strs, base);
                        let len = base.len();
                        let l = cmp::max(0, -1 + *index(&self.ints, l)) as usize;
                        let r = cmp::min(len as Int, *index(&self.ints, r)) as usize;
                        *index_mut(&mut self.strs, res) = base.slice(l, r);
                    }
                    LTFloat(res, l, r) => {
                        let res = *res;
                        let l = *self.get(*l);
                        let r = *self.get(*r);
                        *self.get_mut(res) = (l < r) as Int;
                    }
                    LTInt(res, l, r) => {
                        let res = *res;
                        let l = *self.get(*l);
                        let r = *self.get(*r);
                        *self.get_mut(res) = (l < r) as Int;
                    }
                    LTStr(res, l, r) => {
                        let res = *res;
                        let l = self.get(*l);
                        let r = self.get(*r);
                        *self.get_mut(res) = l.with_str(|l| r.with_str(|r| l < r)) as Int;
                    }
                    GTFloat(res, l, r) => {
                        let res = *res;
                        let l = *self.get(*l);
                        let r = *self.get(*r);
                        *self.get_mut(res) = (l > r) as Int;
                    }
                    GTInt(res, l, r) => {
                        let res = *res;
                        let l = *self.get(*l);
                        let r = *self.get(*r);
                        *self.get_mut(res) = (l > r) as Int;
                    }
                    GTStr(res, l, r) => {
                        let res = *res;
                        let l = self.get(*l);
                        let r = self.get(*r);
                        *self.get_mut(res) = l.with_str(|l| r.with_str(|r| l > r)) as Int;
                    }
                    LTEFloat(res, l, r) => {
                        let res = *res;
                        let l = *self.get(*l);
                        let r = *self.get(*r);
                        *self.get_mut(res) = (l <= r) as Int;
                    }
                    LTEInt(res, l, r) => {
                        let res = *res;
                        let l = *self.get(*l);
                        let r = *self.get(*r);
                        *self.get_mut(res) = (l <= r) as Int;
                    }
                    LTEStr(res, l, r) => {
                        let res = *res;
                        let l = self.get(*l);
                        let r = self.get(*r);
                        *self.get_mut(res) = l.with_str(|l| r.with_str(|r| l <= r)) as Int;
                    }
                    GTEFloat(res, l, r) => {
                        let res = *res;
                        let l = *self.get(*l);
                        let r = *self.get(*r);
                        *self.get_mut(res) = (l >= r) as Int;
                    }
                    GTEInt(res, l, r) => {
                        let res = *res;
                        let l = *self.get(*l);
                        let r = *self.get(*r);
                        *self.get_mut(res) = (l >= r) as Int;
                    }
                    GTEStr(res, l, r) => {
                        let res = *res;
                        let l = self.get(*l);
                        let r = self.get(*r);
                        *self.get_mut(res) = l.with_str(|l| r.with_str(|r| l >= r)) as Int;
                    }
                    EQFloat(res, l, r) => {
                        let res = *res;
                        let l = *self.get(*l);
                        let r = *self.get(*r);
                        *self.get_mut(res) = (l == r) as Int;
                    }
                    EQInt(res, l, r) => {
                        let res = *res;
                        let l = *self.get(*l);
                        let r = *self.get(*r);
                        *self.get_mut(res) = (l == r) as Int;
                    }
                    EQStr(res, l, r) => {
                        let res = *res;
                        let l = self.get(*l);
                        let r = self.get(*r);
                        *self.get_mut(res) = (l == r) as Int;
                    }
                    SetColumn(dst, src) => {
                        let col = *self.get(*dst);
                        if col < 0 {
                            return err!("attempt to access field {}", col);
                        }
                        if col == 0 {
                            self.split_line.clear();
                            self.line.assign_from_str(index(&self.strs, src));
                            self.vars.nf = -1;
                            break cur + 1;
                        }
                        if self.split_line.len() == 0 {
                            self.regexes.split_regex(
                                &self.vars.fs,
                                self.line.as_str(),
                                &mut self.split_line,
                            )?;
                            self.vars.nf = self.split_line.len() as Int;
                        }
                        self.split_line
                            .insert(col as usize - 1, self.get(*src).clone());
                        self.line
                            .assign_from_str(&self.split_line.join(&self.vars.ofs));
                    }
                    GetColumn(dst, src) => {
                        let col = *self.get(*src);
                        let dst = *dst;
                        if col < 0 {
                            return err!("attempt to access field {}", col);
                        }
                        if col == 0 {
                            let line = self.line.as_str().clone();
                            *self.get_mut(dst) = line;
                            break cur + 1;
                        }
                        if self.split_line.len() == 0 {
                            self.regexes.split_regex(
                                &self.vars.fs,
                                self.line.as_str(),
                                &mut self.split_line,
                            )?;
                            self.vars.nf = self.split_line.len() as Int;
                        }
                        let res = self
                            .split_line
                            .get(col as usize - 1)
                            .unwrap_or_else(Default::default);
                        *self.get_mut(dst) = res;
                    }
                    SplitInt(flds, to_split, arr, pat) => {
                        // Index manually here to defeat the borrow checker.
                        let to_split = index(&self.strs, to_split);
                        let arr = index(&self.maps_int_str, arr);
                        let pat = index(&self.strs, pat);
                        let old_len = arr.len();
                        self.regexes.split_regex_intmap(&pat, &to_split, &arr)?;
                        let res = (arr.len() - old_len) as i64;
                        let flds = *flds;
                        *self.get_mut(flds) = res;
                    }
                    SplitStr(flds, to_split, arr, pat) => {
                        // Very similar to above
                        let to_split = index(&self.strs, to_split);
                        let arr = index(&self.maps_str_str, arr);
                        let pat = index(&self.strs, pat);
                        let old_len = arr.len();
                        self.regexes.split_regex_strmap(&pat, &to_split, &arr)?;
                        let res = (arr.len() - old_len) as Int;
                        let flds = *flds;
                        *self.get_mut(flds) = res;
                    }
                    PrintStdout(txt) => {
                        let txt = index(&self.strs, txt);
                        self.write_files.write_str_stdout(txt)?;
                        self.write_files.write_str_stdout(&newline)?;
                    }
                    Print(txt, out, append) => {
                        let txt = index(&self.strs, txt);
                        let out = index(&self.strs, out);
                        self.write_files.write_line(out, txt, *append)?;
                    }
                    Sprintf { dst, fmt, args } => {
                        debug_assert_eq!(scratch.len(), 0);
                        for a in args.iter() {
                            scratch.push(self.format_arg(*a)?);
                        }
                        use runtime::str_impl::DynamicBuf;
                        let fmt_str = index(&self.strs, fmt);
                        let mut buf = DynamicBuf::new(0);
                        fmt_str.with_str(|s| runtime::printf::printf(&mut buf, s, &scratch[..]))?;
                        scratch.clear();
                        let res = unsafe { buf.into_str() };
                        let dst = *dst;
                        *self.get_mut(dst) = res;
                    }
                    Printf { output, fmt, args } => {
                        debug_assert_eq!(scratch.len(), 0);
                        for a in args.iter() {
                            scratch.push(self.format_arg(*a)?);
                        }
                        let fmt_str = index(&self.strs, fmt);
                        if let Some((out_path_reg, append)) = output {
                            let out_path = index(&self.strs, out_path_reg);
                            self.write_files.printf(
                                Some((out_path, *append)),
                                fmt_str,
                                &scratch[..],
                            )
                        } else {
                            // print to stdout.
                            self.write_files.printf(None, fmt_str, &scratch[..])
                        }?;
                        scratch.clear();
                    }
                    Close(file) => {
                        let file = index(&self.strs, file);
                        self.write_files.close(file);
                        self.read_files.close(file);
                    }
                    LookupIntInt(res, arr, k) => {
                        let arr = index(&self.maps_int_int, arr);
                        let k = index(&self.ints, k);
                        let v = arr.get(k).unwrap_or(0);
                        let res = *res;
                        *self.get_mut(res) = v;
                    }
                    LookupIntStr(res, arr, k) => {
                        let arr = index(&self.maps_int_str, arr);
                        let k = index(&self.ints, k);
                        let v = arr.get(k).unwrap_or_else(Default::default);
                        let res = *res;
                        *self.get_mut(res) = v;
                    }
                    LookupIntFloat(res, arr, k) => {
                        let arr = index(&self.maps_int_float, arr);
                        let k = index(&self.ints, k);
                        let v = arr.get(k).unwrap_or(0.0);
                        let res = *res;
                        *self.get_mut(res) = v;
                    }
                    LookupStrInt(res, arr, k) => {
                        let arr = index(&self.maps_str_int, arr);
                        let k = index(&self.strs, k);
                        let v = arr.get(k).unwrap_or(0);
                        let res = *res;
                        *self.get_mut(res) = v;
                    }
                    LookupStrStr(res, arr, k) => {
                        let arr = index(&self.maps_str_str, arr);
                        let k = index(&self.strs, k);
                        let v = arr.get(k).unwrap_or_else(Default::default);
                        let res = *res;
                        *self.get_mut(res) = v;
                    }
                    LookupStrFloat(res, arr, k) => {
                        let arr = index(&self.maps_str_float, arr);
                        let k = index(&self.strs, k);
                        let v = arr.get(k).unwrap_or(0.0);
                        let res = *res;
                        *self.get_mut(res) = v;
                    }
                    ContainsIntInt(res, arr, k) => {
                        let arr = index(&self.maps_int_int, arr);
                        let k = index(&self.ints, k);
                        let v = arr.get(k).is_some() as i64;
                        let res = *res;
                        *self.get_mut(res) = v;
                    }
                    ContainsIntStr(res, arr, k) => {
                        let arr = index(&self.maps_int_str, arr);
                        let k = index(&self.ints, k);
                        let v = arr.get(k).is_some() as i64;
                        let res = *res;
                        *self.get_mut(res) = v;
                    }
                    ContainsIntFloat(res, arr, k) => {
                        let arr = index(&self.maps_int_float, arr);
                        let k = index(&self.ints, k);
                        let v = arr.get(k).is_some() as i64;
                        let res = *res;
                        *self.get_mut(res) = v;
                    }
                    ContainsStrInt(res, arr, k) => {
                        let arr = index(&self.maps_str_int, arr);
                        let k = index(&self.strs, k);
                        let v = arr.get(k).is_some() as i64;
                        let res = *res;
                        *self.get_mut(res) = v;
                    }
                    ContainsStrStr(res, arr, k) => {
                        let arr = index(&self.maps_str_str, arr);
                        let k = index(&self.strs, k);
                        let v = arr.get(k).is_some() as i64;
                        let res = *res;
                        *self.get_mut(res) = v;
                    }
                    ContainsStrFloat(res, arr, k) => {
                        let arr = index(&self.maps_str_float, arr);
                        let k = index(&self.strs, k);
                        let v = arr.get(k).is_some() as i64;
                        let res = *res;
                        *self.get_mut(res) = v;
                    }
                    DeleteIntInt(arr, k) => {
                        let arr = index(&self.maps_int_int, arr);
                        let k = index(&self.ints, k);
                        arr.delete(k);
                    }
                    DeleteIntFloat(arr, k) => {
                        let arr = index(&self.maps_int_float, arr);
                        let k = index(&self.ints, k);
                        arr.delete(k);
                    }
                    DeleteIntStr(arr, k) => {
                        let arr = index(&self.maps_int_str, arr);
                        let k = index(&self.ints, k);
                        arr.delete(k);
                    }
                    DeleteStrInt(arr, k) => {
                        let arr = index(&self.maps_str_int, arr);
                        let k = index(&self.strs, k);
                        arr.delete(k);
                    }
                    DeleteStrFloat(arr, k) => {
                        let arr = index(&self.maps_str_float, arr);
                        let k = index(&self.strs, k);
                        arr.delete(k);
                    }
                    DeleteStrStr(arr, k) => {
                        let arr = index(&self.maps_str_str, arr);
                        let k = index(&self.strs, k);
                        arr.delete(k);
                    }
                    LenIntInt(res, arr) => {
                        let arr = *arr;
                        let len = self.get(arr).len();
                        let res = *res;
                        *self.get_mut(res) = len as Int;
                    }
                    LenIntFloat(res, arr) => {
                        let arr = *arr;
                        let len = self.get(arr).len();
                        let res = *res;
                        *self.get_mut(res) = len as Int;
                    }
                    LenIntStr(res, arr) => {
                        let arr = *arr;
                        let len = self.get(arr).len();
                        let res = *res;
                        *self.get_mut(res) = len as Int;
                    }
                    LenStrInt(res, arr) => {
                        let arr = *arr;
                        let len = self.get(arr).len();
                        let res = *res;
                        *self.get_mut(res) = len as Int;
                    }
                    LenStrFloat(res, arr) => {
                        let arr = *arr;
                        let len = self.get(arr).len();
                        let res = *res;
                        *self.get_mut(res) = len as Int;
                    }
                    LenStrStr(res, arr) => {
                        let arr = *arr;
                        let len = self.get(arr).len();
                        let res = *res;
                        *self.get_mut(res) = len as Int;
                    }
                    StoreIntInt(arr, k, v) => {
                        let arr = index(&self.maps_int_int, arr);
                        let k = index(&self.ints, k).clone();
                        let v = index(&self.ints, v).clone();
                        arr.insert(k, v);
                    }
                    StoreIntFloat(arr, k, v) => {
                        let arr = index(&self.maps_int_float, arr);
                        let k = index(&self.ints, k).clone();
                        let v = index(&self.floats, v).clone();
                        arr.insert(k, v);
                    }
                    StoreIntStr(arr, k, v) => {
                        let arr = index(&self.maps_int_str, arr);
                        let k = index(&self.ints, k).clone();
                        let v = index(&self.strs, v).clone();
                        arr.insert(k, v);
                    }
                    StoreStrInt(arr, k, v) => {
                        let arr = index(&self.maps_str_int, arr);
                        let k = index(&self.strs, k).clone();
                        let v = index(&self.ints, v).clone();
                        arr.insert(k, v);
                    }
                    StoreStrFloat(arr, k, v) => {
                        let arr = index(&self.maps_str_float, arr);
                        let k = index(&self.strs, k).clone();
                        let v = index(&self.floats, v).clone();
                        arr.insert(k, v);
                    }
                    StoreStrStr(arr, k, v) => {
                        let arr = index(&self.maps_str_str, arr);
                        let k = index(&self.strs, k).clone();
                        let v = index(&self.strs, v).clone();
                        arr.insert(k, v);
                    }
                    LoadVarStr(dst, var) => {
                        let s = self.vars.load_str(*var)?;
                        let dst = *dst;
                        *self.get_mut(dst) = s;
                    }
                    StoreVarStr(var, src) => {
                        let src = *src;
                        let s = self.get(src).clone();
                        self.vars.store_str(*var, s)?;
                    }
                    LoadVarInt(dst, var) => {
                        if *var == NF && self.split_line.len() == 0 {
                            self.regexes.split_regex(
                                &self.vars.fs,
                                self.line.as_str(),
                                &mut self.split_line,
                            )?;
                            self.vars.nf = self.split_line.len() as Int;
                        }
                        let i = self.vars.load_int(*var)?;
                        let dst = *dst;
                        *self.get_mut(dst) = i;
                    }
                    StoreVarInt(var, src) => {
                        let src = *src;
                        let s = *self.get(src);
                        self.vars.store_int(*var, s)?;
                    }
                    LoadVarIntMap(dst, var) => {
                        let arr = self.vars.load_intmap(*var)?;
                        let dst = *dst;
                        *self.get_mut(dst) = arr;
                    }
                    StoreVarIntMap(var, src) => {
                        let src = *src;
                        let s = self.get(src).clone();
                        self.vars.store_intmap(*var, s)?;
                    }
                    IterBeginIntInt(dst, arr) => {
                        let arr = *arr;
                        let iter = self.get(arr).to_iter();
                        let dst = *dst;
                        *self.get_mut(dst) = iter;
                    }
                    IterBeginIntFloat(dst, arr) => {
                        let arr = *arr;
                        let iter = self.get(arr).to_iter();
                        let dst = *dst;
                        *self.get_mut(dst) = iter;
                    }
                    IterBeginIntStr(dst, arr) => {
                        let arr = *arr;
                        let iter = self.get(arr).to_iter();
                        let dst = *dst;
                        *self.get_mut(dst) = iter;
                    }
                    IterBeginStrInt(dst, arr) => {
                        let arr = *arr;
                        let iter = self.get(arr).to_iter();
                        let dst = *dst;
                        *self.get_mut(dst) = iter;
                    }
                    IterBeginStrFloat(dst, arr) => {
                        let arr = *arr;
                        let iter = self.get(arr).to_iter();
                        let dst = *dst;
                        *self.get_mut(dst) = iter;
                    }
                    IterBeginStrStr(dst, arr) => {
                        let arr = *arr;
                        let iter = self.get(arr).to_iter();
                        let dst = *dst;
                        *self.get_mut(dst) = iter;
                    }
                    IterHasNextInt(dst, iter) => {
                        let res = self.get(*iter).has_next() as Int;
                        let dst = *dst;
                        *self.get_mut(dst) = res;
                    }
                    IterHasNextStr(dst, iter) => {
                        let res = self.get(*iter).has_next() as Int;
                        let dst = *dst;
                        *self.get_mut(dst) = res;
                    }
                    IterGetNextInt(dst, iter) => {
                        let res = unsafe { self.get(*iter).get_next().clone() };
                        let dst = *dst;
                        *self.get_mut(dst) = res;
                    }
                    IterGetNextStr(dst, iter) => {
                        let res = unsafe { self.get(*iter).get_next().clone() };
                        let dst = *dst;
                        *self.get_mut(dst) = res;
                    }
                    MovInt(dst, src) => {
                        let src = *src;
                        let src_contents = self.get(src).clone();
                        let dst = *dst;
                        *self.get_mut(dst) = src_contents;
                    }
                    MovFloat(dst, src) => {
                        let src = *src;
                        let src_contents = self.get(src).clone();
                        let dst = *dst;
                        *self.get_mut(dst) = src_contents;
                    }
                    MovStr(dst, src) => {
                        let src = *src;
                        let src_contents = self.get(src).clone();
                        let dst = *dst;
                        *self.get_mut(dst) = src_contents;
                    }
                    MovMapIntInt(dst, src) => {
                        let src = *src;
                        let src_contents = self.get(src).clone();
                        let dst = *dst;
                        *self.get_mut(dst) = src_contents;
                    }
                    MovMapIntFloat(dst, src) => {
                        let src = *src;
                        let src_contents = self.get(src).clone();
                        let dst = *dst;
                        *self.get_mut(dst) = src_contents;
                    }
                    MovMapIntStr(dst, src) => {
                        let src = *src;
                        let src_contents = self.get(src).clone();
                        let dst = *dst;
                        *self.get_mut(dst) = src_contents;
                    }
                    MovMapStrInt(dst, src) => {
                        let src = *src;
                        let src_contents = self.get(src).clone();
                        let dst = *dst;
                        *self.get_mut(dst) = src_contents;
                    }
                    MovMapStrFloat(dst, src) => {
                        let src = *src;
                        let src_contents = self.get(src).clone();
                        let dst = *dst;
                        *self.get_mut(dst) = src_contents;
                    }
                    MovMapStrStr(dst, src) => {
                        let src = *src;
                        let src_contents = self.get(src).clone();
                        let dst = *dst;
                        *self.get_mut(dst) = src_contents;
                    }
                    // TODO add error logging for these errors perhaps?
                    ReadErr(dst, file) => {
                        let dst = *dst;
                        let file = index(&self.strs, file);
                        let res = self.read_files.read_err(file)?;
                        *self.get_mut(dst) = res;
                    }
                    NextLine(dst, file) => {
                        let dst = *dst;
                        let file = index(&self.strs, file);
                        match self
                            .regexes
                            .get_line(file, &self.vars.rs, &mut self.read_files)
                        {
                            Ok(l) => *self.get_mut(dst) = l,
                            Err(_) => *self.get_mut(dst) = "".into(),
                        };
                    }
                    ReadErrStdin(dst) => {
                        let dst = *dst;
                        let res = self.read_files.read_err_stdin();
                        *self.get_mut(dst) = res;
                    }
                    NextLineStdin(dst) => {
                        let dst = *dst;
                        let res = self
                            .regexes
                            .get_line_stdin(&self.vars.rs, &mut self.read_files)?;
                        *self.get_mut(dst) = res;
                    }
                    JmpIf(cond, lbl) => {
                        let cond = *cond;
                        if *self.get(cond) != 0 {
                            break lbl.0 as usize;
                        }
                    }
                    Jmp(lbl) => {
                        break lbl.0 as usize;
                    }
                    PushInt(reg) => {
                        let reg = *reg;
                        self.push(reg)
                    }
                    PushFloat(reg) => {
                        let reg = *reg;
                        self.push(reg)
                    }
                    PushStr(reg) => {
                        let reg = *reg;
                        self.push(reg)
                    }
                    PushIntInt(reg) => {
                        let reg = *reg;
                        self.push(reg)
                    }
                    PushIntFloat(reg) => {
                        let reg = *reg;
                        self.push(reg)
                    }
                    PushIntStr(reg) => {
                        let reg = *reg;
                        self.push(reg)
                    }
                    PushStrInt(reg) => {
                        let reg = *reg;
                        self.push(reg)
                    }
                    PushStrFloat(reg) => {
                        let reg = *reg;
                        self.push(reg)
                    }
                    PushStrStr(reg) => {
                        let reg = *reg;
                        self.push(reg)
                    }

                    PopInt(reg) => {
                        let reg = *reg;
                        self.pop(reg)
                    }
                    PopFloat(reg) => {
                        let reg = *reg;
                        self.pop(reg)
                    }
                    PopStr(reg) => {
                        let reg = *reg;
                        self.pop(reg)
                    }
                    PopIntInt(reg) => {
                        let reg = *reg;
                        self.pop(reg)
                    }
                    PopIntFloat(reg) => {
                        let reg = *reg;
                        self.pop(reg)
                    }
                    PopIntStr(reg) => {
                        let reg = *reg;
                        self.pop(reg)
                    }
                    PopStrInt(reg) => {
                        let reg = *reg;
                        self.pop(reg)
                    }
                    PopStrFloat(reg) => {
                        let reg = *reg;
                        self.pop(reg)
                    }
                    PopStrStr(reg) => {
                        let reg = *reg;
                        self.pop(reg)
                    }
                    Call(func) => {
                        self.stack.push((cur_fn, Label(cur + 1)));
                        cur_fn = *func;
                        instrs = &mut self.instrs[*func];
                        break 0;
                    }
                    Ret => {
                        if let Some((func, Label(inst))) = self.stack.pop() {
                            cur_fn = func;
                            instrs = &mut self.instrs[func];
                            break inst as usize;
                        } else {
                            break 'outer Ok(());
                        }
                    }
                    Halt => break 'outer Ok(()),
                };
                break cur + 1;
            };
        }
    }
}

// TODO: Add a pass that does checking of indexes once.
// That could justify no checking during interpretation.
#[cfg(debug_assertions)]
const CHECKED: bool = true;
#[cfg(not(debug_assertions))]
const CHECKED: bool = false;

#[inline(always)]
pub(crate) fn index<'a, T>(Storage { regs, .. }: &'a Storage<T>, reg: &Reg<T>) -> &'a T {
    if CHECKED {
        &regs[reg.index()]
    } else {
        debug_assert!(reg.index() < regs.len());
        unsafe { regs.get_unchecked(reg.index()) }
    }
}

#[inline(always)]
pub(crate) fn index_mut<'a, T>(
    Storage { regs, .. }: &'a mut Storage<T>,
    reg: &Reg<T>,
) -> &'a mut T {
    if CHECKED {
        &mut regs[reg.index()]
    } else {
        debug_assert!(reg.index() < regs.len());
        unsafe { regs.get_unchecked_mut(reg.index()) }
    }
}

#[inline(always)]
pub(crate) fn push<'a, T: Clone>(s: &'a mut Storage<T>, reg: &Reg<T>) {
    let v = index(s, reg).clone();
    s.stack.push(v);
}

#[inline(always)]
pub(crate) fn pop<'a, T: Clone>(s: &'a mut Storage<T>) -> T {
    s.stack.pop().expect("pop must be called on nonempty stack")
}

// Used in benchmarking code.

#[cfg(test)]
impl<T: Default> Storage<T> {
    fn reset(&mut self) {
        self.stack.clear();
        for i in self.regs.iter_mut() {
            *i = Default::default();
        }
    }
}

#[cfg(test)]
impl<'a, LR: LineReader> Interp<'a, LR> {
    pub(crate) fn reset(&mut self) {
        self.stack = Default::default();
        self.vars = Default::default();
        self.line.assign_from_str(&Default::default());
        self.split_line = LazyVec::new();
        self.regexes = Default::default();
        self.floats.reset();
        self.ints.reset();
        self.strs.reset();
        self.maps_int_int.reset();
        self.maps_int_float.reset();
        self.maps_int_str.reset();
        self.maps_str_int.reset();
        self.maps_str_float.reset();
        self.maps_str_str.reset();
        self.iters_int.reset();
        self.iters_str.reset();
    }
}
