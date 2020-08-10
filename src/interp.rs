use crate::builtins::Variable;
use crate::bytecode::{Get, Instr, Label, Pop, Reg};
use crate::common::{NumTy, Result, Stage};
use crate::compile::{self, Ty};
use crate::pushdown::FieldSet;
use crate::runtime::{self, Float, Int, Line, LineReader, Str, UniqueStr};

use crossbeam::scope;
use hashbrown::HashMap;
use rand::{self, rngs::StdRng, Rng, SeedableRng};

use std::cmp;
use std::mem;

type ClassicReader = runtime::splitter::regex::RegexSplitter<Box<dyn std::io::Read>>;

#[derive(Default)]
pub(crate) struct Storage<T> {
    pub(crate) regs: Vec<T>,
    pub(crate) stack: Vec<T>,
}

/// Core represents a subset of runtime structures that are relevant to both the bytecode
/// interpreter and the compiled runtimes.
pub(crate) struct Core<'a> {
    pub vars: runtime::Variables<'a>,
    pub regexes: runtime::RegexCache,
    pub write_files: runtime::FileWrite,
    pub rng: StdRng,
    pub current_seed: u64,
    pub slots: Slots,
}

#[derive(Default)]
pub(crate) struct Slots {
    pub int: Vec<Int>,
    pub float: Vec<Float>,
    pub strs: Vec<UniqueStr<'static>>,
    pub intint: Vec<HashMap<Int, Int>>,
    pub intfloat: Vec<HashMap<Int, Float>>,
    pub intstr: Vec<HashMap<Int, UniqueStr<'static>>>,
    pub strint: Vec<HashMap<UniqueStr<'static>, Int>>,
    pub strfloat: Vec<HashMap<UniqueStr<'static>, Float>>,
    pub strstr: Vec<HashMap<UniqueStr<'static>, UniqueStr<'static>>>,
}

/// A Simple helper trait for implement aggregations for slot values and variables.
trait Agg {
    fn agg(self, other: Self) -> Self;
}
impl Agg for Int {
    fn agg(self, other: Int) -> Int {
        self + other
    }
}
impl Agg for Float {
    fn agg(self, other: Float) -> Float {
        self + other
    }
}
impl<'a> Agg for UniqueStr<'a> {
    fn agg(self, other: UniqueStr<'a>) -> UniqueStr<'a> {
        let slf = self.into_str();
        let otr = other.into_str();
        UniqueStr::from(Str::concat(slf, otr))
    }
}
impl<K: std::hash::Hash + Eq, V: Agg + Default> Agg for HashMap<K, V> {
    fn agg(mut self, other: HashMap<K, V>) -> HashMap<K, V> {
        for (k, v) in other {
            let entry = self.entry(k).or_insert(Default::default());
            let v2 = mem::replace(entry, Default::default());
            *entry = v2.agg(v);
        }
        self
    }
}

/// StageResult is a Send subset of Core that can be extracted for inter-stage aggregation in a
/// parallel script.
pub(crate) struct StageResult {
    slots: Slots,
    // TODO: put more variables in here?
    nr: Int,
}

impl Slots {
    // TODO: Concatenation of UniqueStr could be faster if we implemented N-way combine.
    fn combine(&mut self, mut other: Slots) {
        macro_rules! for_each_slot_pair {
            ($s1:ident, $s2:ident, $body:expr) => {
                for_each_slot_pair!(
                    $s1, $s2, $body, int, float, strs, intint, intfloat, intstr, strint, strfloat,
                    strstr
                );
            };
            ($s1:ident, $s2:ident, $body:expr, $($fld:tt),*) => {$({
                let $s1 = &mut self.$fld;
                let $s2 = &mut other.$fld;
                $body
            });*};
        }

        for_each_slot_pair!(a, b, {
            a.resize_with(std::cmp::max(a.len(), b.len()), Default::default);
            for (a_elt, b_elt_v) in a.iter_mut().zip(b.drain(..)) {
                let a_elt_v = mem::replace(a_elt, Default::default());
                *a_elt = a_elt_v.agg(b_elt_v);
            }
        });
    }
}

pub fn set_slot<T: Default>(vec: &mut Vec<T>, slot: usize, v: T) {
    if slot < vec.len() {
        vec[slot] = v;
        return;
    }
    vec.resize_with(slot, Default::default);
    vec.push(v)
}

pub fn combine_slot<T: Default>(vec: &mut Vec<T>, slot: usize, f: impl FnOnce(T) -> T) {
    if slot < vec.len() {
        let res = f(std::mem::replace(&mut vec[slot], Default::default()));
        vec[slot] = res;
        return;
    }
    vec.resize_with(slot, Default::default);
    let res = f(Default::default());
    vec.push(res)
}

impl<'a> Core<'a> {
    pub fn from_writers(fw: runtime::FileWrite) -> Core<'a> {
        let seed: u64 = rand::thread_rng().gen();
        Core {
            vars: Default::default(),
            regexes: Default::default(),
            write_files: fw,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            current_seed: seed,
            slots: Default::default(),
        }
    }
    pub fn new(ff: impl runtime::writers::FileFactory) -> Core<'a> {
        let seed: u64 = rand::thread_rng().gen();
        Core {
            vars: Default::default(),
            regexes: Default::default(),
            write_files: runtime::FileWrite::new(ff),
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            current_seed: seed,
            slots: Default::default(),
        }
    }

    pub fn extract_result(&mut self) -> StageResult {
        StageResult {
            slots: mem::replace(&mut self.slots, Default::default()),
            nr: self.vars.nr,
        }
    }

    pub fn combine(&mut self, StageResult { slots, nr }: StageResult) {
        self.slots.combine(slots);
        self.vars.nr = self.vars.nr.agg(nr);
    }

    pub fn reseed(&mut self, seed: u64) -> u64 /* old seed */ {
        self.rng = StdRng::seed_from_u64(seed);
        let old_seed = self.current_seed;
        self.current_seed = seed;
        old_seed
    }

    pub fn reseed_random(&mut self) -> u64 /* old seed */ {
        self.reseed(rand::thread_rng().gen::<u64>())
    }

    pub fn match_regex(&mut self, s: &Str<'a>, pat: &Str<'a>) -> Result<Int> {
        self.regexes.regex_match_loc(&mut self.vars, pat, s)
    }

    pub fn is_match_regex(&mut self, s: &Str<'a>, pat: &Str<'a>) -> Result<bool> {
        self.regexes.is_regex_match(pat, s)
    }

    pub fn load_int(&mut self, slot: usize) -> Int {
        self.slots.int[slot]
    }
    pub fn load_float(&mut self, slot: usize) -> Float {
        self.slots.float[slot]
    }
    pub fn load_str(&mut self, slot: usize) -> Str<'a> {
        mem::replace(&mut self.slots.strs[slot], Default::default())
            .into_str()
            .upcast()
    }
    pub fn load_intint(&mut self, slot: usize) -> runtime::IntMap<Int> {
        mem::replace(&mut self.slots.intint[slot], Default::default()).into()
    }
    pub fn load_intfloat(&mut self, slot: usize) -> runtime::IntMap<Float> {
        mem::replace(&mut self.slots.intfloat[slot], Default::default()).into()
    }
    pub fn load_intstr(&mut self, slot: usize) -> runtime::IntMap<Str<'a>> {
        mem::replace(&mut self.slots.intstr[slot], Default::default())
            .into_iter()
            .map(|(k, v)| (k, v.into_str().upcast()))
            .collect()
    }
    pub fn load_strint(&mut self, slot: usize) -> runtime::StrMap<'a, Int> {
        mem::replace(&mut self.slots.strint[slot], Default::default())
            .into_iter()
            .map(|(k, v)| (k.into_str().upcast(), v))
            .collect()
    }
    pub fn load_strfloat(&mut self, slot: usize) -> runtime::StrMap<'a, Float> {
        mem::replace(&mut self.slots.strfloat[slot], Default::default())
            .into_iter()
            .map(|(k, v)| (k.into_str().upcast(), v))
            .collect()
    }
    pub fn load_strstr(&mut self, slot: usize) -> runtime::StrMap<'a, Str<'a>> {
        mem::replace(&mut self.slots.strstr[slot], Default::default())
            .into_iter()
            .map(|(k, v)| (k.into_str().upcast(), v.into_str().upcast()))
            .collect()
    }

    pub fn store_int(&mut self, slot: usize, i: Int) {
        set_slot(&mut self.slots.int, slot, i)
    }
    pub fn store_float(&mut self, slot: usize, f: Float) {
        set_slot(&mut self.slots.float, slot, f)
    }
    pub fn store_str(&mut self, slot: usize, s: Str<'a>) {
        set_slot(&mut self.slots.strs, slot, s.unmoor().into())
    }
    pub fn store_intint(&mut self, slot: usize, s: runtime::IntMap<Int>) {
        set_slot(
            &mut self.slots.intint,
            slot,
            s.iter(|i| i.map(|(k, v)| (*k, *v)).collect()),
        )
    }
    pub fn store_intfloat(&mut self, slot: usize, s: runtime::IntMap<Float>) {
        set_slot(
            &mut self.slots.intfloat,
            slot,
            s.iter(|i| i.map(|(k, v)| (*k, *v)).collect()),
        )
    }
    pub fn store_intstr(&mut self, slot: usize, s: runtime::IntMap<Str<'a>>) {
        set_slot(
            &mut self.slots.intstr,
            slot,
            s.iter(|i| i.map(|(k, v)| (*k, v.clone().unmoor().into())).collect()),
        )
    }
    pub fn store_strint(&mut self, slot: usize, s: runtime::StrMap<'a, Int>) {
        set_slot(
            &mut self.slots.strint,
            slot,
            s.iter(|i| i.map(|(k, v)| (k.clone().unmoor().into(), *v)).collect()),
        )
    }
    pub fn store_strfloat(&mut self, slot: usize, s: runtime::StrMap<'a, Float>) {
        set_slot(
            &mut self.slots.strfloat,
            slot,
            s.iter(|i| i.map(|(k, v)| (k.clone().unmoor().into(), *v)).collect()),
        )
    }
    pub fn store_strstr(&mut self, slot: usize, s: runtime::StrMap<'a, Str<'a>>) {
        set_slot(
            &mut self.slots.strstr,
            slot,
            s.iter(|i| {
                i.map(|(k, v)| (k.clone().unmoor().into(), v.clone().unmoor().into()))
                    .collect()
            }),
        )
    }
}

pub(crate) struct Interp<'a, LR: LineReader = ClassicReader> {
    // index of `instrs` that contains "main"
    main_func: Stage<usize>,
    num_workers: usize,
    instrs: Vec<Vec<Instr<'a>>>,
    stack: Vec<(usize /*function*/, Label /*instr*/)>,

    line: LR::Line,
    read_files: runtime::FileRead<LR>,

    core: Core<'a>,

    // Core storage.
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

impl<'a, LR: LineReader> Interp<'a, LR> {
    pub(crate) fn new(
        instrs: Vec<Vec<Instr<'a>>>,
        main_func: Stage<usize>,
        num_workers: usize,
        regs: impl Fn(compile::Ty) -> usize,
        stdin: LR,
        ff: impl runtime::writers::FileFactory,
        used_fields: &FieldSet,
    ) -> Self {
        use compile::Ty::*;
        Interp {
            main_func,
            num_workers,
            instrs,
            stack: Default::default(),
            floats: default_of(regs(Float)),
            ints: default_of(regs(Int)),
            strs: default_of(regs(Str)),
            core: Core::new(ff),

            line: Default::default(),
            read_files: runtime::FileRead::new(stdin, used_fields),

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

    fn reset_file_vars(&mut self) {
        self.core.vars.fnr = 0;
        self.core.vars.filename = self.read_files.stdin_filename().upcast();
    }

    pub(crate) fn run_parallel(&mut self) -> Result<()> {
        if self.num_workers <= 1 {
            return self.run_serial();
        }
        let mut handles = self.read_files.try_resize(self.num_workers);
        if handles.len() <= 1 {
            return self.run_serial();
        }
        let (begin, middle, end) = match self.main_func {
            Stage::Par {
                begin,
                main_loop,
                end,
            } => (begin, main_loop, end),
            Stage::Main(_) => {
                return err!("unexpected Main-only configuration for parallel execution")
            }
        };
        let main_loop = if let Some(main_loop) = middle {
            main_loop
        } else {
            return self.run_serial();
        };
        if let Some(off) = begin {
            self.run_at(off)?;
        }

        let mut initial_fr = handles.pop().unwrap()();
        mem::swap(&mut initial_fr, &mut self.read_files);
        let mut results: Vec<StageResult> = Vec::with_capacity(handles.len() + 1);
        fn wrap_error<T, S>(r: std::result::Result<Result<T>, S>) -> Result<T> {
            match r {
                Ok(Ok(t)) => Ok(t),
                Ok(Err(e)) => Err(e),
                Err(_) => err!("error in executing worker thread"),
            }
        }
        let scope_res = scope(|s| {
            let mut join_handles = Vec::with_capacity(handles.len() + 1);
            let float_size = self.floats.regs.len();
            let ints_size = self.ints.regs.len();
            let strs_size = self.strs.regs.len();
            let maps_int_int_size = self.maps_int_int.regs.len();
            let maps_int_float_size = self.maps_int_float.regs.len();
            let maps_int_str_size = self.maps_int_str.regs.len();
            let maps_str_int_size = self.maps_str_int.regs.len();
            let maps_str_float_size = self.maps_str_float.regs.len();
            let maps_str_str_size = self.maps_str_str.regs.len();
            let iters_int_size = self.iters_int.regs.len();
            let iters_str_size = self.iters_str.regs.len();
            for handle in handles.into_iter() {
                let writers = self.core.write_files.clone();
                let instrs = self.instrs.clone();
                join_handles.push(s.spawn(move |_| {
                    let mut interp = Interp {
                        main_func: Stage::Main(main_loop),
                        num_workers: 1,
                        instrs,
                        stack: Default::default(),
                        core: Core::from_writers(writers),
                        line: Default::default(),
                        read_files: handle(),

                        floats: default_of(float_size),
                        ints: default_of(ints_size),
                        strs: default_of(strs_size),
                        maps_int_int: default_of(maps_int_int_size),
                        maps_int_float: default_of(maps_int_float_size),
                        maps_int_str: default_of(maps_int_str_size),
                        maps_str_int: default_of(maps_str_int_size),
                        maps_str_float: default_of(maps_str_float_size),
                        maps_str_str: default_of(maps_str_str_size),
                        iters_int: default_of(iters_int_size),
                        iters_str: default_of(iters_str_size),
                    };
                    interp.run_at(main_loop)?;
                    Ok(interp.core.extract_result())
                }));
            }
            for h in join_handles.into_iter() {
                // TODO: see what we can make of any panic errors here. It'd be nice to return something
                // more sensible.

                results.push(wrap_error(h.join())?);
            }
            Ok(())
        });
        wrap_error(scope_res)?;
        for r in results.into_iter() {
            self.core.combine(r);
        }
        if let Some(end) = end {
            self.run_at(end)?;
        }
        Ok(())
    }

    pub(crate) fn run_serial(&mut self) -> Result<()> {
        let offs: crate::smallvec::SmallVec<[usize; 3]> = self.main_func.iter().cloned().collect();
        for off in offs.into_iter() {
            self.run_at(off)?
        }
        Ok(())
    }

    pub(crate) fn run(&mut self) -> Result<()> {
        // TODO make this another param.
        match self.main_func {
            Stage::Main(_) => self.run_serial(),
            Stage::Par { .. } => self.run_parallel(),
        }
    }

    pub(crate) fn run_at(&mut self, mut cur_fn: usize) -> Result<()> {
        use Instr::*;
        let mut scratch: Vec<runtime::FormatArg> = Vec::new();
        // We are only accessing one vector at a time here, but it's hard to convince the borrow
        // checker of this fact, so we access the vectors through raw pointers.
        let mut instrs = (&mut self.instrs[cur_fn]) as *mut Vec<Instr<'a>>;
        let mut cur = 0;

        // Local macros help to eliminate boilerplate in type-specific instructions.
        // StoreSlot<T>(ident, slot_id)
        macro_rules! store_slot {
            ($src: ident, $slot_id: ident, $slot_meth:tt) => {{
                let src = *$src;
                let new_val = self.get(src).clone();
                self.core.$slot_meth(*$slot_id as usize, new_val);
            }};
        }
        // LoadSlot<T>(ident, slot_id)
        macro_rules! load_slot {
            ($dst: ident, $slot_id: ident, $slot_meth:tt) => {{
                let dst = *$dst;
                let new_val = self.core.$slot_meth(*$slot_id as usize);
                *self.get_mut(dst) = new_val;
            }};
        }
        // IterBegin<T>(dst, arr)
        macro_rules! iter_begin {
            ($dst:ident, $arr:ident) => {{
                let arr = *$arr;
                let dst = *$dst;
                let iter = self.get(arr).to_iter();
                *self.get_mut(dst) = iter;
            }};
        }
        'outer: loop {
            // must end with Halt
            cur = loop {
                debug_assert!(cur < unsafe { (*instrs).len() });
                use Variable::*;
                match unsafe { (*instrs).get_unchecked(cur) } {
                    StoreConstStr(sr, s) => {
                        let sr = *sr;
                        *self.get_mut(sr) = s.clone_str()
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
                    HexStrToInt(ir, sr) => {
                        let i = self.get(*sr).with_str(runtime::hextoi);
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
                    Pow(res, l, r) => {
                        let res = *res;
                        let l = *self.get(*l);
                        let r = *self.get(*r);
                        *self.get_mut(res) = l.powf(r);
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
                    Float1(ff, dst, src) => {
                        let f = *index(&self.floats, src);
                        let dst = *dst;
                        *self.get_mut(dst) = ff.eval1(f);
                    }
                    Float2(ff, dst, x, y) => {
                        let fx = *index(&self.floats, x);
                        let fy = *index(&self.floats, y);
                        let dst = *dst;
                        *self.get_mut(dst) = ff.eval2(fx, fy);
                    }
                    Rand(dst) => {
                        let res: f64 = self.core.rng.gen_range(0.0, 1.0);
                        *index_mut(&mut self.floats, dst) = res;
                    }
                    Srand(res, seed) => {
                        let old_seed = self.core.reseed(*index(&self.ints, seed) as u64);
                        *index_mut(&mut self.ints, res) = old_seed as Int;
                    }
                    ReseedRng(res) => {
                        *index_mut(&mut self.ints, res) = self.core.reseed_random() as Int;
                    }
                    Concat(res, l, r) => {
                        let res = *res;
                        let l = self.get(*l).clone();
                        let r = self.get(*r).clone();
                        *self.get_mut(res) = Str::concat(l, r);
                    }
                    Match(res, l, r) => {
                        *index_mut(&mut self.ints, res) = self
                            .core
                            .match_regex(index(&self.strs, l), index(&self.strs, r))?;
                    }
                    IsMatch(res, l, r) => {
                        *index_mut(&mut self.ints, res) = self
                            .core
                            .is_match_regex(index(&self.strs, l), index(&self.strs, r))?
                            as Int;
                    }
                    SubstrIndex(res, s, t) => {
                        let res = *res;
                        let s = index(&self.strs, s);
                        let t = index(&self.strs, t);
                        *self.get_mut(res) = s
                            .with_str(|s| t.with_str(|t| s.find(t).map(|x| x + 1).unwrap_or(0)))
                            as Int;
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
                            self.core
                                .regexes
                                .with_regex(pat, |re| in_s.subst_first(re, s))?
                        };
                        *index_mut(&mut self.strs, in_s) = subbed;
                        *index_mut(&mut self.ints, res) = new as Int;
                    }
                    GSub(res, pat, s, in_s) => {
                        let (subbed, subs_made) = {
                            let pat = index(&self.strs, pat);
                            let s = index(&self.strs, s);
                            let in_s = index(&self.strs, in_s);
                            self.core
                                .regexes
                                .with_regex(pat, |re| in_s.subst_all(re, s))?
                        };
                        *index_mut(&mut self.strs, in_s) = subbed;
                        *index_mut(&mut self.ints, res) = subs_made;
                    }
                    EscapeCSV(res, s) => {
                        *index_mut(&mut self.strs, res) = {
                            let s = index(&self.strs, s);
                            runtime::escape_csv(s)
                        };
                    }
                    EscapeTSV(res, s) => {
                        *index_mut(&mut self.strs, res) = {
                            let s = index(&self.strs, s);
                            runtime::escape_tsv(s)
                        };
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
                        let v = index(&self.strs, src);
                        self.line
                            .set_col(col, v, &self.core.vars.ofs, &mut self.core.regexes)?;
                    }
                    GetColumn(dst, src) => {
                        let col = *self.get(*src);
                        let dst = *dst;
                        let res = self.line.get_col(
                            col,
                            &self.core.vars.fs,
                            &self.core.vars.ofs,
                            &mut self.core.regexes,
                        )?;
                        *self.get_mut(dst) = res;
                    }
                    JoinCSV(dst, start, end) => {
                        let nf = self.line.nf(&self.core.vars.fs, &mut self.core.regexes)?;
                        *index_mut(&mut self.strs, dst) = {
                            let start = *index(&self.ints, start);
                            let end = *index(&self.ints, end);
                            self.line.join_cols(start, end, &",".into(), nf, |s| {
                                runtime::escape_csv(&s)
                            })?
                        };
                    }
                    JoinTSV(dst, start, end) => {
                        let nf = self.line.nf(&self.core.vars.fs, &mut self.core.regexes)?;
                        *index_mut(&mut self.strs, dst) = {
                            let start = *index(&self.ints, start);
                            let end = *index(&self.ints, end);
                            self.line.join_cols(start, end, &"\t".into(), nf, |s| {
                                runtime::escape_tsv(&s)
                            })?
                        };
                    }
                    JoinColumns(dst, start, end, sep) => {
                        let nf = self.line.nf(&self.core.vars.fs, &mut self.core.regexes)?;
                        *index_mut(&mut self.strs, dst) = {
                            let sep = index(&self.strs, sep);
                            let start = *index(&self.ints, start);
                            let end = *index(&self.ints, end);
                            self.line.join_cols(start, end, sep, nf, |s| s)?
                        };
                    }
                    SplitInt(flds, to_split, arr, pat) => {
                        // Index manually here to defeat the borrow checker.
                        let to_split = index(&self.strs, to_split);
                        let arr = index(&self.maps_int_str, arr);
                        let pat = index(&self.strs, pat);
                        let old_len = arr.len();
                        self.core
                            .regexes
                            .split_regex_intmap(&pat, &to_split, &arr)?;
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
                        self.core
                            .regexes
                            .split_regex_strmap(&pat, &to_split, &arr)?;
                        let res = (arr.len() - old_len) as Int;
                        let flds = *flds;
                        *self.get_mut(flds) = res;
                    }
                    PrintStdout(txt) => {
                        let txt = index(&self.strs, txt);
                        // Why do this? We want to exit cleanly when output is closed. We use this
                        // pattern for other IO functions as well.
                        if let Err(_) = self.core.write_files.write_str_stdout(txt) {
                            return Ok(());
                        }
                    }
                    Print(txt, out, append) => {
                        let txt = index(&self.strs, txt);
                        let out = index(&self.strs, out);
                        if let Err(_) = self.core.write_files.write_str(out, txt, *append) {
                            return Ok(());
                        };
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
                        let res = if let Some((out_path_reg, append)) = output {
                            let out_path = index(&self.strs, out_path_reg);
                            self.core.write_files.printf(
                                Some((out_path, *append)),
                                fmt_str,
                                &scratch[..],
                            )
                        } else {
                            // print to stdout.
                            self.core.write_files.printf(None, fmt_str, &scratch[..])
                        };
                        if res.is_err() {
                            return Ok(());
                        }
                        scratch.clear();
                    }
                    Close(file) => {
                        let file = index(&self.strs, file);
                        // NB this may create an unused entry in write_files. It would not be
                        // terribly difficult to optimize the close path to include an existence
                        // check first.
                        self.core.write_files.close(file)?;
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
                        let s = self.core.vars.load_str(*var)?;
                        let dst = *dst;
                        *self.get_mut(dst) = s;
                    }
                    StoreVarStr(var, src) => {
                        let src = *src;
                        let s = self.get(src).clone();
                        self.core.vars.store_str(*var, s)?;
                    }
                    LoadVarInt(dst, var) => {
                        // If someone explicitly sets NF to a different value, this means we will
                        // ignore it. I think that is fine.
                        if let NF = *var {
                            self.core.vars.nf =
                                self.line.nf(&self.core.vars.fs, &mut self.core.regexes)? as Int;
                        }
                        let i = self.core.vars.load_int(*var)?;
                        let dst = *dst;
                        *self.get_mut(dst) = i;
                    }
                    StoreVarInt(var, src) => {
                        let src = *src;
                        let s = *self.get(src);
                        self.core.vars.store_int(*var, s)?;
                    }
                    LoadVarIntMap(dst, var) => {
                        let arr = self.core.vars.load_intmap(*var)?;
                        let dst = *dst;
                        *self.get_mut(dst) = arr;
                    }
                    StoreVarIntMap(var, src) => {
                        let src = *src;
                        let s = self.get(src).clone();
                        self.core.vars.store_intmap(*var, s)?;
                    }

                    LoadSlotInt(dst, slot) => load_slot!(dst, slot, load_int),
                    LoadSlotFloat(dst, slot) => load_slot!(dst, slot, load_float),
                    LoadSlotStr(dst, slot) => load_slot!(dst, slot, load_str),
                    LoadSlotIntInt(dst, slot) => load_slot!(dst, slot, load_intint),
                    LoadSlotIntFloat(dst, slot) => load_slot!(dst, slot, load_intfloat),
                    LoadSlotIntStr(dst, slot) => load_slot!(dst, slot, load_intstr),
                    LoadSlotStrInt(dst, slot) => load_slot!(dst, slot, load_strint),
                    LoadSlotStrFloat(dst, slot) => load_slot!(dst, slot, load_strfloat),
                    LoadSlotStrStr(dst, slot) => load_slot!(dst, slot, load_strstr),
                    StoreSlotInt(src, slot) => store_slot!(src, slot, store_int),
                    StoreSlotFloat(src, slot) => store_slot!(src, slot, store_float),
                    StoreSlotStr(src, slot) => store_slot!(src, slot, store_str),
                    StoreSlotIntInt(src, slot) => store_slot!(src, slot, store_intint),
                    StoreSlotIntFloat(src, slot) => store_slot!(src, slot, store_intfloat),
                    StoreSlotIntStr(src, slot) => store_slot!(src, slot, store_intstr),
                    StoreSlotStrInt(src, slot) => store_slot!(src, slot, store_strint),
                    StoreSlotStrFloat(src, slot) => store_slot!(src, slot, store_strfloat),
                    StoreSlotStrStr(src, slot) => store_slot!(src, slot, store_strstr),

                    IterBeginIntInt(dst, arr) => iter_begin!(dst, arr),
                    IterBeginIntFloat(dst, arr) => iter_begin!(dst, arr),
                    IterBeginIntStr(dst, arr) => iter_begin!(dst, arr),
                    IterBeginStrInt(dst, arr) => iter_begin!(dst, arr),
                    IterBeginStrFloat(dst, arr) => iter_begin!(dst, arr),
                    IterBeginStrStr(dst, arr) => iter_begin!(dst, arr),

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
                    AllocMapIntInt(dst) => {
                        let dst = *dst;
                        *self.get_mut(dst) = Default::default();
                    }
                    AllocMapIntFloat(dst) => {
                        let dst = *dst;
                        *self.get_mut(dst) = Default::default();
                    }
                    AllocMapIntStr(dst) => {
                        let dst = *dst;
                        *self.get_mut(dst) = Default::default();
                    }
                    AllocMapStrInt(dst) => {
                        let dst = *dst;
                        *self.get_mut(dst) = Default::default();
                    }
                    AllocMapStrFloat(dst) => {
                        let dst = *dst;
                        *self.get_mut(dst) = Default::default();
                    }
                    AllocMapStrStr(dst) => {
                        let dst = *dst;
                        *self.get_mut(dst) = Default::default();
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
                        match self.core.regexes.get_line(
                            file,
                            &self.core.vars.rs,
                            &mut self.read_files,
                        ) {
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
                        let (changed, res) = self
                            .core
                            .regexes
                            .get_line_stdin(&self.core.vars.rs, &mut self.read_files)?;
                        if changed {
                            self.reset_file_vars();
                        }
                        *self.get_mut(dst) = res;
                    }
                    NextLineStdinFused() => {
                        let changed = self.core.regexes.get_line_stdin_reuse(
                            &self.core.vars.rs,
                            &mut self.read_files,
                            &mut self.line,
                        )?;
                        if changed {
                            self.reset_file_vars()
                        }
                    }
                    NextFile() => {
                        self.read_files.next_file()?;
                        self.reset_file_vars();
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

pub(crate) fn push<'a, T: Clone>(s: &'a mut Storage<T>, reg: &Reg<T>) {
    let v = index(s, reg).clone();
    s.stack.push(v);
}

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
        self.core.vars = Default::default();
        self.line = Default::default();
        self.core.regexes = Default::default();
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
