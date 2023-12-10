use crate::builtins::Variable;
use crate::bytecode::{Get, Instr, Label, Reg};
use crate::common::{NumTy, Result, Stage};
use crate::compile::{self, Ty};
use crate::pushdown::FieldSet;
use crate::runtime::{self, Float, Int, Line, LineReader, Str, UniqueStr};

use crossbeam::scope;
use crossbeam_channel::bounded;
use hashbrown::HashMap;
use rand::{self, rngs::StdRng, Rng, SeedableRng};
use regex::bytes::Regex;

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

impl<'a> Drop for Core<'a> {
    fn drop(&mut self) {
        if let Err(e) = self.write_files.shutdown() {
            eprintln_ignore!("{}", e);
        }
    }
}

/// Slots are used for transmitting data across different "stages" of a parallel computation. In
/// order to send data to and from worker threads, we load and store them into dynamically-sized
/// "slots". These aren't normal registers, because slots store `Send` variants of the frawk
/// runtime types; making the value safe for sending between threads may involve performing a
/// deep copy.
#[derive(Default, Clone)]
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
        // Strings are not aggregated explicitly.
        if other.is_empty() {
            self
        } else {
            other
        }
    }
}
impl<K: std::hash::Hash + Eq, V: Agg + Default> Agg for HashMap<K, V> {
    fn agg(mut self, other: HashMap<K, V>) -> HashMap<K, V> {
        for (k, v) in other {
            let entry = self.entry(k).or_default();
            let v2 = mem::take(entry);
            *entry = v2.agg(v);
        }
        self
    }
}

/// StageResult is a Send subset of Core that can be extracted for inter-stage aggregation in a
/// parallel script.
pub(crate) struct StageResult {
    slots: Slots,
    // TODO: put more variables in here? Most builtin variables are just going to be propagated
    // from the initial thread.
    nr: Int,
    rc: i32,
}

impl Slots {
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
                let a_elt_v = mem::take(a_elt);
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
        let res = f(std::mem::take(&mut vec[slot]));
        vec[slot] = res;
        return;
    }
    vec.resize_with(slot, Default::default);
    let res = f(Default::default());
    vec.push(res)
}

impl<'a> Core<'a> {
    pub fn shuttle(&self, pid: Int) -> impl FnOnce() -> Core<'a> + Send {
        use crate::builtins::Variables;
        let seed: u64 = rand::thread_rng().gen();
        let fw = self.write_files.clone();
        let fs: UniqueStr<'a> = self.vars.fs.clone().into();
        let ofs: UniqueStr<'a> = self.vars.ofs.clone().into();
        let rs: UniqueStr<'a> = self.vars.rs.clone().into();
        let ors: UniqueStr<'a> = self.vars.ors.clone().into();
        let filename: UniqueStr<'a> = self.vars.filename.clone().into();
        let argv = self.vars.argv.shuttle();
        let fi = self.vars.fi.shuttle();
        let slots = self.slots.clone();
        move || {
            let vars = Variables {
                fs: fs.into_str(),
                ofs: ofs.into_str(),
                ors: ors.into_str(),
                rs: rs.into_str(),
                filename: filename.into_str(),
                pid,
                nf: 0,
                nr: 0,
                fnr: 0,
                rstart: 0,
                rlength: 0,
                argc: 0,
                argv: argv.into(),
                fi: fi.into(),
            };
            Core {
                vars,
                regexes: Default::default(),
                write_files: fw,
                rng: rand::rngs::StdRng::seed_from_u64(seed),
                current_seed: seed,
                slots,
            }
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

    pub fn extract_result(&mut self, rc: i32) -> StageResult {
        StageResult {
            slots: mem::take(&mut self.slots),
            nr: self.vars.nr,
            rc,
        }
    }

    pub fn combine(&mut self, StageResult { slots, nr, rc: _ }: StageResult) {
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

    pub fn match_const_regex(&mut self, s: &Str<'a>, pat: &Regex) -> Result<Int> {
        runtime::RegexCache::regex_const_match_loc(&mut self.vars, pat, s)
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
        mem::take(&mut self.slots.strs[slot]).into_str().upcast()
    }
    pub fn load_intint(&mut self, slot: usize) -> runtime::IntMap<Int> {
        mem::take(&mut self.slots.intint[slot]).into()
    }
    pub fn load_intfloat(&mut self, slot: usize) -> runtime::IntMap<Float> {
        mem::take(&mut self.slots.intfloat[slot]).into()
    }
    pub fn load_intstr(&mut self, slot: usize) -> runtime::IntMap<Str<'a>> {
        mem::take(&mut self.slots.intstr[slot])
            .into_iter()
            .map(|(k, v)| (k, v.into_str().upcast()))
            .collect()
    }
    pub fn load_strint(&mut self, slot: usize) -> runtime::StrMap<'a, Int> {
        mem::take(&mut self.slots.strint[slot])
            .into_iter()
            .map(|(k, v)| (k.into_str().upcast(), v))
            .collect()
    }
    pub fn load_strfloat(&mut self, slot: usize) -> runtime::StrMap<'a, Float> {
        mem::take(&mut self.slots.strfloat[slot])
            .into_iter()
            .map(|(k, v)| (k.into_str().upcast(), v))
            .collect()
    }
    pub fn load_strstr(&mut self, slot: usize) -> runtime::StrMap<'a, Str<'a>> {
        mem::take(&mut self.slots.strstr[slot])
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

macro_rules! map_regs {
    ($map_ty:expr, $map_reg:ident, $body:expr) => {{
        let _placeholder_k = 0u32;
        let _placeholder_v = 0u32;
        map_regs!($map_ty, $map_reg, _placeholder_k, _placeholder_v, $body)
    }};
    ($map_ty:expr, $map_reg:ident, $key_reg:ident, $val_reg:ident, $body:expr) => {{
        let _placeholder_iter = 0u32;
        map_regs!(
            $map_ty,
            $map_reg,
            $key_reg,
            $val_reg,
            _placeholder_iter,
            $body
        )
    }};
    ($map_ty:expr, $map_reg:ident, $key_reg:ident, $val_reg:ident, $iter_reg:ident, $body:expr) => {{
        let map_ty = $map_ty;
        match map_ty {
            Ty::MapIntInt => {
                let $map_reg: Reg<runtime::IntMap<Int>> = $map_reg.into();
                let $key_reg: Reg<Int> = $key_reg.into();
                let $val_reg: Reg<Int> = $val_reg.into();
                let $iter_reg: Reg<runtime::Iter<Int>> = $iter_reg.into();
                $body
            }
            Ty::MapIntFloat => {
                let $map_reg: Reg<runtime::IntMap<Float>> = $map_reg.into();
                let $key_reg: Reg<Int> = $key_reg.into();
                let $val_reg: Reg<Float> = $val_reg.into();
                let $iter_reg: Reg<runtime::Iter<Int>> = $iter_reg.into();
                $body
            }
            Ty::MapIntStr => {
                let $map_reg: Reg<runtime::IntMap<Str<'a>>> = $map_reg.into();
                let $key_reg: Reg<Int> = $key_reg.into();
                let $val_reg: Reg<Str<'a>> = $val_reg.into();
                let $iter_reg: Reg<runtime::Iter<Int>> = $iter_reg.into();
                $body
            }
            Ty::MapStrInt => {
                let $map_reg: Reg<runtime::StrMap<'a, Int>> = $map_reg.into();
                let $key_reg: Reg<Str<'a>> = $key_reg.into();
                let $val_reg: Reg<Int> = $val_reg.into();
                let $iter_reg: Reg<runtime::Iter<Str<'a>>> = $iter_reg.into();
                $body
            }
            Ty::MapStrFloat => {
                let $map_reg: Reg<runtime::StrMap<'a, Float>> = $map_reg.into();
                let $key_reg: Reg<Str<'a>> = $key_reg.into();
                let $val_reg: Reg<Float> = $val_reg.into();
                let $iter_reg: Reg<runtime::Iter<Str<'a>>> = $iter_reg.into();
                $body
            }
            Ty::MapStrStr => {
                let $map_reg: Reg<runtime::StrMap<'a, Str<'a>>> = $map_reg.into();
                let $key_reg: Reg<Str<'a>> = $key_reg.into();
                let $val_reg: Reg<Str<'a>> = $val_reg.into();
                let $iter_reg: Reg<runtime::Iter<Str<'a>>> = $iter_reg.into();
                $body
            }
            Ty::Null | Ty::Int | Ty::Float | Ty::Str | Ty::IterInt | Ty::IterStr => panic!(
                "attempting to perform map operations on non-map type: {:?}",
                map_ty
            ),
        }
    }};
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
        named_columns: Option<Vec<&[u8]>>,
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
            read_files: runtime::FileRead::new(stdin, used_fields.clone(), named_columns),

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
            Ty::Int => (*self.get(Reg::<Int>::from(reg))).into(),
            Ty::Float => (*self.get(Reg::<Float>::from(reg))).into(),
            Ty::Null => runtime::FormatArg::Null,
            _ => return err!("non-scalar (s)printf argument type {:?}", ty),
        })
    }

    fn reset_file_vars(&mut self) {
        self.core.vars.fnr = 0;
        self.core.vars.filename = self.read_files.stdin_filename().upcast();
    }

    pub(crate) fn run_parallel(&mut self) -> Result<i32> {
        if self.num_workers <= 1 {
            return self.run_serial();
        }
        let handles = self.read_files.try_resize(self.num_workers - 1);
        if handles.is_empty() {
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
            let rc = self.run_at(off)?;
            if rc != 0 {
                return Ok(rc);
            }
        }
        if self.core.write_files.flush_stdout().is_err() {
            return Ok(1);
        }
        // For handling the worker portion, we want to transfer the current stdin progress to a
        // worker thread, but to withhold any progress on other files open for read. We'll swap
        // these back in when we execute the `end` block, if there is one.
        let mut old_read_files = mem::take(&mut self.read_files.inputs);
        fn wrap_error<T, S>(r: std::result::Result<Result<T>, S>) -> Result<T> {
            match r {
                Ok(Ok(t)) => Ok(t),
                Ok(Err(e)) => Err(e),
                Err(_) => err!("error in executing worker thread"),
            }
        }
        let scope_res = scope(|s| {
            let (sender, receiver) = bounded(handles.len());
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
            for (i, handle) in handles.into_iter().enumerate() {
                let sender = sender.clone();
                let core_shuttle = self.core.shuttle(i as Int + 2);
                let instrs = self.instrs.clone();
                s.spawn(move |_| {
                    if let Some(read_files) = handle() {
                        let mut interp = Interp {
                            main_func: Stage::Main(main_loop),
                            num_workers: 1,
                            instrs,
                            stack: Default::default(),
                            core: core_shuttle(),
                            line: Default::default(),
                            read_files,

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
                        let res = interp.run_at(main_loop);

                        // Ignore errors, as it means another thread executed with an error and we are
                        // exiting anyway.
                        let _ = match res {
                            Err(e) => sender.send(Err(e)),
                            Ok(rc) => sender.send(Ok(interp.core.extract_result(rc))),
                        };
                    }
                });
            }
            mem::drop(sender);
            self.core.vars.pid = 1;
            let mut rc = self.run_at(main_loop)?;
            self.core.vars.pid = 0;
            while let Ok(res) = receiver.recv() {
                let res = res?;
                let sub_rc = res.rc;
                self.core.combine(res);
                if rc == 0 && sub_rc != 0 {
                    rc = sub_rc;
                }
            }
            Ok(rc)
        });
        let rc = wrap_error(scope_res)?;
        if rc != 0 {
            return Ok(rc);
        }
        if let Some(end) = end {
            mem::swap(&mut self.read_files.inputs, &mut old_read_files);
            Ok(self.run_at(end)?)
        } else {
            Ok(0)
        }
    }

    pub(crate) fn run_serial(&mut self) -> Result<i32> {
        let offs: smallvec::SmallVec<[usize; 3]> = self.main_func.iter().cloned().collect();
        for off in offs.into_iter() {
            let rc = self.run_at(off)?;
            if rc != 0 {
                return Ok(rc);
            }
        }
        Ok(0)
    }

    pub(crate) fn run(&mut self) -> Result<i32> {
        match self.main_func {
            Stage::Main(_) => self.run_serial(),
            Stage::Par { .. } => self.run_parallel(),
        }
    }

    #[allow(clippy::never_loop)]
    pub(crate) fn run_at(&mut self, mut cur_fn: usize) -> Result<i32> {
        use Instr::*;
        let mut scratch: Vec<runtime::FormatArg> = Vec::new();
        // We are only accessing one vector at a time here, but it's hard to convince the borrow
        // checker of this fact, so we access the vectors through raw pointers.
        let mut instrs = (&mut self.instrs[cur_fn]) as *mut Vec<Instr<'a>>;
        let mut cur = 0;

        'outer: loop {
            // This somewhat ersatz structure is to allow 'cur' to be reassigned
            // in most but not all branches in the big match below.
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
                        let i = self.get(*sr).with_bytes(runtime::hextoi);
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
                        let is_empty = self.get(sr).with_bytes(|bs| bs.is_empty());
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
                    Int1(bw, dst, src) => {
                        let i = *index(&self.ints, src);
                        let dst = *dst;
                        *self.get_mut(dst) = bw.eval1(i);
                    }
                    Int2(bw, dst, x, y) => {
                        let ix = *index(&self.ints, x);
                        let iy = *index(&self.ints, y);
                        let dst = *dst;
                        *self.get_mut(dst) = bw.eval2(ix, iy);
                    }
                    Rand(dst) => {
                        let res: f64 = self.core.rng.gen_range(0.0..=1.0);
                        *index_mut(&mut self.floats, dst) = res;
                    }
                    Srand(res, seed) => {
                        let old_seed = self.core.reseed(*index(&self.ints, seed) as u64);
                        *index_mut(&mut self.ints, res) = old_seed as Int;
                    }
                    ReseedRng(res) => {
                        *index_mut(&mut self.ints, res) = self.core.reseed_random() as Int;
                    }
                    StartsWithConst(res, s, bs) => {
                        let s_bytes = unsafe { &*index(&self.strs, s).get_bytes() };
                        *index_mut(&mut self.ints, res) =
                            (bs.len() <= s_bytes.len() && s_bytes[..bs.len()] == **bs) as Int;
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
                    MatchConst(res, x, pat) => {
                        *index_mut(&mut self.ints, res) =
                            runtime::RegexCache::regex_const_match(pat, index(&self.strs, x))
                                as Int;
                    }
                    IsMatchConst(res, x, pat) => {
                        *index_mut(&mut self.ints, res) =
                            self.core.match_const_regex(index(&self.strs, x), pat)?;
                    }
                    SubstrIndex(res, s, t) => {
                        let res = *res;
                        let s = index(&self.strs, s);
                        let t = index(&self.strs, t);
                        *self.get_mut(res) = runtime::string_search::index_substr(t, s);
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
                    GenSubDynamic(res, pat, s, how, in_s) => {
                        let subbed = {
                            let pat = index(&self.strs, pat);
                            let s = index(&self.strs, s);
                            let how = index(&self.strs, how);
                            let in_s = index(&self.strs, in_s);
                            self.core
                                .regexes
                                .with_regex(pat, |re| in_s.gen_subst_dynamic(re, s, how))?
                        };
                        *index_mut(&mut self.strs, res) = subbed;
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
                        let l = cmp::max(0, -1 + *index(&self.ints, l));
                        *index_mut(&mut self.strs, res) = if l as usize >= len {
                            Str::default()
                        } else {
                            let r = cmp::min(len as Int, l.saturating_add(*index(&self.ints, r)))
                                as usize;
                            base.slice(l as usize, r)
                        };
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
                        *self.get_mut(res) = l.with_bytes(|l| r.with_bytes(|r| l < r)) as Int;
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
                        *self.get_mut(res) = l.with_bytes(|l| r.with_bytes(|r| l > r)) as Int;
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
                        *self.get_mut(res) = l.with_bytes(|l| r.with_bytes(|r| l <= r)) as Int;
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
                        *self.get_mut(res) = l.with_bytes(|l| r.with_bytes(|r| l >= r)) as Int;
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
                    ToUpperAscii(dst, src) => {
                        let res = index(&self.strs, src).to_upper_ascii();
                        *index_mut(&mut self.strs, dst) = res;
                    }
                    ToLowerAscii(dst, src) => {
                        let res = index(&self.strs, src).to_lower_ascii();
                        *index_mut(&mut self.strs, dst) = res;
                    }
                    SplitInt(flds, to_split, arr, pat) => {
                        // Index manually here to defeat the borrow checker.
                        let to_split = index(&self.strs, to_split);
                        let arr = index(&self.maps_int_str, arr);
                        let pat = index(&self.strs, pat);
                        self.core.regexes.split_regex_intmap(pat, to_split, arr)?;
                        let res = arr.len() as Int;
                        let flds = *flds;
                        *self.get_mut(flds) = res;
                    }
                    SplitStr(flds, to_split, arr, pat) => {
                        // Very similar to above
                        let to_split = index(&self.strs, to_split);
                        let arr = index(&self.maps_str_str, arr);
                        let pat = index(&self.strs, pat);
                        self.core.regexes.split_regex_strmap(pat, to_split, arr)?;
                        let res = arr.len() as Int;
                        let flds = *flds;
                        *self.get_mut(flds) = res;
                    }
                    Sprintf { dst, fmt, args } => {
                        debug_assert_eq!(scratch.len(), 0);
                        for a in args.iter() {
                            scratch.push(self.format_arg(*a)?);
                        }
                        use runtime::str_impl::DynamicBuf;
                        let fmt_str = index(&self.strs, fmt);
                        let mut buf = DynamicBuf::new(0);
                        fmt_str
                            .with_bytes(|bs| runtime::printf::printf(&mut buf, bs, &scratch[..]))?;
                        scratch.clear();
                        let res = buf.into_str();
                        let dst = *dst;
                        *self.get_mut(dst) = res;
                    }
                    PrintAll { output, args } => {
                        let mut scratch_strs =
                            smallvec::SmallVec::<[&Str; 4]>::with_capacity(args.len());
                        for a in args {
                            scratch_strs.push(index(&self.strs, a));
                        }
                        let res = if let Some((out_path_reg, fspec)) = output {
                            let out_path = index(&self.strs, out_path_reg);
                            self.core
                                .write_files
                                .write_all(&scratch_strs[..], Some((out_path, *fspec)))
                        } else {
                            self.core.write_files.write_all(&scratch_strs[..], None)
                        };
                        if res.is_err() {
                            return Ok(0);
                        }
                    }
                    Printf { output, fmt, args } => {
                        debug_assert_eq!(scratch.len(), 0);
                        for a in args.iter() {
                            scratch.push(self.format_arg(*a)?);
                        }
                        let fmt_str = index(&self.strs, fmt);
                        let res = if let Some((out_path_reg, fspec)) = output {
                            let out_path = index(&self.strs, out_path_reg);
                            self.core.write_files.printf(
                                Some((out_path, *fspec)),
                                fmt_str,
                                &scratch[..],
                            )
                        } else {
                            // print to stdout.
                            self.core.write_files.printf(None, fmt_str, &scratch[..])
                        };
                        if res.is_err() {
                            return Ok(0);
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
                    RunCmd(dst, cmd) => {
                        *index_mut(&mut self.ints, dst) =
                            index(&self.strs, cmd).with_bytes(runtime::run_command);
                    }
                    Exit(code) => return Ok(*index(&self.ints, code) as i32),
                    Lookup {
                        map_ty,
                        dst,
                        map,
                        key,
                    } => self.lookup(*map_ty, *dst, *map, *key),
                    Contains {
                        map_ty,
                        dst,
                        map,
                        key,
                    } => self.contains(*map_ty, *dst, *map, *key),
                    Delete { map_ty, map, key } => self.delete(*map_ty, *map, *key),
                    Clear { map_ty, map } => self.clear(*map_ty, *map),
                    Len { map_ty, map, dst } => self.len(*map_ty, *map, *dst),
                    Store {
                        map_ty,
                        map,
                        key,
                        val,
                    } => self.store_map(*map_ty, *map, *key, *val),
                    IncInt {
                        map_ty,
                        map,
                        key,
                        by,
                        dst,
                    } => self.inc_map_int(*map_ty, *map, *key, *by, *dst),
                    IncFloat {
                        map_ty,
                        map,
                        key,
                        by,
                        dst,
                    } => self.inc_map_float(*map_ty, *map, *key, *by, *dst),
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
                    LoadVarStrMap(dst, var) => {
                        let arr = self.core.vars.load_strmap(*var)?;
                        let dst = *dst;
                        *self.get_mut(dst) = arr;
                    }
                    StoreVarStrMap(var, src) => {
                        let src = *src;
                        let s = self.get(src).clone();
                        self.core.vars.store_strmap(*var, s)?;
                    }

                    IterBegin { map_ty, map, dst } => self.iter_begin(*map_ty, *map, *dst),
                    IterHasNext { iter_ty, dst, iter } => self.iter_has_next(*iter_ty, *dst, *iter),
                    IterGetNext { iter_ty, dst, iter } => self.iter_get_next(*iter_ty, *dst, *iter),

                    LoadSlot { ty, dst, slot } => self.load_slot(*ty, *dst, *slot),
                    StoreSlot { ty, src, slot } => self.store_slot(*ty, *src, *slot),
                    Mov(ty, dst, src) => self.mov(*ty, *dst, *src),
                    AllocMap(ty, reg) => self.alloc_map(*ty, *reg),

                    // TODO add error logging for these errors perhaps?
                    ReadErr(dst, file, is_file) => {
                        let dst = *dst;
                        let file = index(&self.strs, file);
                        let res = if *is_file {
                            self.read_files.read_err(file)?
                        } else {
                            self.read_files.read_err_cmd(file)?
                        };
                        *self.get_mut(dst) = res;
                    }
                    NextLine(dst, file, is_file) => {
                        let dst = *dst;
                        let file = index(&self.strs, file);
                        match self.core.regexes.get_line(
                            file,
                            &self.core.vars.rs,
                            &mut self.read_files,
                            *is_file,
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
                    UpdateUsedFields() => {
                        let fi = &self.core.vars.fi;
                        self.read_files.update_named_columns(fi);
                    }
                    SetFI(key, val) => {
                        let key = *index(&self.ints, key);
                        let val = *index(&self.ints, val);
                        let col = self.line.get_col(
                            key,
                            &self.core.vars.fs,
                            &self.core.vars.ofs,
                            &mut self.core.regexes,
                        )?;
                        self.core.vars.fi.insert(col, val);
                    }
                    JmpIf(cond, lbl) => {
                        let cond = *cond;
                        if *self.get(cond) != 0 {
                            break lbl.0;
                        }
                    }
                    Jmp(lbl) => {
                        break lbl.0;
                    }
                    Push(ty, reg) => self.push_reg(*ty, *reg),
                    Pop(ty, reg) => self.pop_reg(*ty, *reg),
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
                            break inst;
                        } else {
                            break 'outer Ok(0);
                        }
                    }
                };
                break cur + 1;
            };
        }
    }
    fn mov(&mut self, ty: Ty, dst: NumTy, src: NumTy) {
        match ty {
            Ty::Int => {
                let src = *index(&self.ints, &src.into());
                *index_mut(&mut self.ints, &dst.into()) = src;
            }
            Ty::Float => {
                let src = *index(&self.floats, &src.into());
                *index_mut(&mut self.floats, &dst.into()) = src;
            }
            Ty::Str => {
                let src = index(&self.strs, &src.into()).clone();
                *index_mut(&mut self.strs, &dst.into()) = src;
            }
            Ty::MapIntInt => {
                let src = index(&self.maps_int_int, &src.into()).clone();
                *index_mut(&mut self.maps_int_int, &dst.into()) = src;
            }
            Ty::MapIntFloat => {
                let src = index(&self.maps_int_float, &src.into()).clone();
                *index_mut(&mut self.maps_int_float, &dst.into()) = src;
            }
            Ty::MapIntStr => {
                let src = index(&self.maps_int_str, &src.into()).clone();
                *index_mut(&mut self.maps_int_str, &dst.into()) = src;
            }
            Ty::MapStrInt => {
                let src = index(&self.maps_str_int, &src.into()).clone();
                *index_mut(&mut self.maps_str_int, &dst.into()) = src;
            }
            Ty::MapStrFloat => {
                let src = index(&self.maps_str_float, &src.into()).clone();
                *index_mut(&mut self.maps_str_float, &dst.into()) = src;
            }
            Ty::MapStrStr => {
                let src = index(&self.maps_str_str, &src.into()).clone();
                *index_mut(&mut self.maps_str_str, &dst.into()) = src;
            }
            Ty::Null | Ty::IterInt | Ty::IterStr => {
                panic!("invalid type for move operation: {:?}", ty)
            }
        }
    }
    fn alloc_map(&mut self, ty: Ty, reg: NumTy) {
        map_regs!(ty, reg, *self.get_mut(reg) = Default::default())
    }
    fn lookup(&mut self, map_ty: Ty, dst: NumTy, map: NumTy, key: NumTy) {
        map_regs!(map_ty, map, key, dst, {
            let res = self.get(map).get(self.get(key));
            *self.get_mut(dst) = res;
        });
    }
    fn contains(&mut self, map_ty: Ty, dst: NumTy, map: NumTy, key: NumTy) {
        let _v = 0u32;
        let dst: Reg<Int> = dst.into();
        map_regs!(map_ty, map, key, _v, {
            let res = self.get(map).contains(self.get(key)) as Int;
            *self.get_mut(dst) = res;
        });
    }
    fn delete(&mut self, map_ty: Ty, map: NumTy, key: NumTy) {
        let _v = 0u32;
        map_regs!(map_ty, map, key, _v, {
            self.get(map).delete(self.get(key))
        });
    }
    fn clear(&mut self, map_ty: Ty, map: NumTy) {
        map_regs!(map_ty, map, self.get(map).clear());
    }

    // Allowing this because it allows for easier use of the map_regs macro.
    #[allow(clippy::clone_on_copy)]
    fn store_map(&mut self, map_ty: Ty, map: NumTy, key: NumTy, val: NumTy) {
        map_regs!(map_ty, map, key, val, {
            let k = self.get(key).clone();
            let v = self.get(val).clone();
            self.get(map).insert(k, v);
        });
    }
    fn inc_map_int(&mut self, map_ty: Ty, map: NumTy, key: NumTy, by: Reg<Int>, dst: NumTy) {
        map_regs!(map_ty, map, key, dst, {
            let k = self.get(key);
            let m = self.get(map);
            let by = *self.get(by);
            let res = m.inc_int(k, by);
            *self.get_mut(dst) = res;
        })
    }
    fn inc_map_float(&mut self, map_ty: Ty, map: NumTy, key: NumTy, by: Reg<Float>, dst: NumTy) {
        map_regs!(map_ty, map, key, dst, {
            let k = self.get(key);
            let m = self.get(map);
            let by = *self.get(by);
            let res = m.inc_float(k, by);
            *self.get_mut(dst) = res;
        })
    }
    fn len(&mut self, map_ty: Ty, map: NumTy, dst: NumTy) {
        let len = map_regs!(map_ty, map, self.get(map).len() as Int);
        *index_mut(&mut self.ints, &dst.into()) = len;
    }
    fn iter_begin(&mut self, map_ty: Ty, map: NumTy, dst: NumTy) {
        let _k = 0u32;
        let _v = 0u32;
        map_regs!(map_ty, map, _k, _v, dst, {
            let iter = self.get(map).to_iter();
            *self.get_mut(dst) = iter;
        })
    }
    fn iter_has_next(&mut self, iter_ty: Ty, dst: NumTy, iter: NumTy) {
        match iter_ty {
            Ty::IterInt => {
                let res = index(&self.iters_int, &iter.into()).has_next() as Int;
                *index_mut(&mut self.ints, &dst.into()) = res;
            }
            Ty::IterStr => {
                let res = index(&self.iters_str, &iter.into()).has_next() as Int;
                *index_mut(&mut self.ints, &dst.into()) = res;
            }
            x => panic!("non-iterator type passed to has_next: {:?}", x),
        }
    }
    fn iter_get_next(&mut self, iter_ty: Ty, dst: NumTy, iter: NumTy) {
        match iter_ty {
            Ty::IterInt => {
                let res = unsafe { *index(&self.iters_int, &iter.into()).get_next() };
                *index_mut(&mut self.ints, &dst.into()) = res;
            }
            Ty::IterStr => {
                let res = unsafe { index(&self.iters_str, &iter.into()).get_next().clone() };
                *index_mut(&mut self.strs, &dst.into()) = res;
            }
            x => panic!("non-iterator type passed to get_next: {:?}", x),
        }
    }
    fn load_slot(&mut self, ty: Ty, dst: NumTy, slot: Int) {
        let slot = slot as usize;
        macro_rules! do_load {
            ($load_meth:tt, $reg_fld:tt) => {
                *index_mut(&mut self.$reg_fld, &dst.into()) = self.core.$load_meth(slot)
            };
        }
        match ty {
            Ty::Int => do_load!(load_int, ints),
            Ty::Float => do_load!(load_float, floats),
            Ty::Str => do_load!(load_str, strs),
            Ty::MapIntInt => do_load!(load_intint, maps_int_int),
            Ty::MapIntFloat => do_load!(load_intfloat, maps_int_float),
            Ty::MapIntStr => do_load!(load_intstr, maps_int_str),
            Ty::MapStrInt => do_load!(load_strint, maps_str_int),
            Ty::MapStrFloat => do_load!(load_strfloat, maps_str_float),
            Ty::MapStrStr => do_load!(load_strstr, maps_str_str),
            Ty::Null | Ty::IterInt | Ty::IterStr => {
                panic!("unexpected operand type to slot operation: {:?}", ty)
            }
        }
    }
    fn store_slot(&mut self, ty: Ty, src: NumTy, slot: Int) {
        let slot = slot as usize;
        macro_rules! do_store {
            ($store_meth:tt, $reg_fld:tt) => {
                self.core
                    .$store_meth(slot, index(&self.$reg_fld, &src.into()).clone())
            };
        }
        match ty {
            Ty::Int => do_store!(store_int, ints),
            Ty::Float => do_store!(store_float, floats),
            Ty::Str => do_store!(store_str, strs),
            Ty::MapIntInt => do_store!(store_intint, maps_int_int),
            Ty::MapIntFloat => do_store!(store_intfloat, maps_int_float),
            Ty::MapIntStr => do_store!(store_intstr, maps_int_str),
            Ty::MapStrInt => do_store!(store_strint, maps_str_int),
            Ty::MapStrFloat => do_store!(store_strfloat, maps_str_float),
            Ty::MapStrStr => do_store!(store_strstr, maps_str_str),
            Ty::Null | Ty::IterInt | Ty::IterStr => panic!("unsupported slot type: {:?}", ty),
        }
    }
    fn push_reg(&mut self, ty: Ty, src: NumTy) {
        match ty {
            Ty::Int => push(&mut self.ints, &src.into()),
            Ty::Float => push(&mut self.floats, &src.into()),
            Ty::Str => push(&mut self.strs, &src.into()),
            Ty::MapIntInt => push(&mut self.maps_int_int, &src.into()),
            Ty::MapIntFloat => push(&mut self.maps_int_float, &src.into()),
            Ty::MapIntStr => push(&mut self.maps_int_str, &src.into()),
            Ty::MapStrInt => push(&mut self.maps_str_int, &src.into()),
            Ty::MapStrFloat => push(&mut self.maps_str_float, &src.into()),
            Ty::MapStrStr => push(&mut self.maps_str_str, &src.into()),
            Ty::Null | Ty::IterInt | Ty::IterStr => {
                panic!("unsupported register type for push operation: {:?}", ty)
            }
        }
    }
    fn pop_reg(&mut self, ty: Ty, dst: NumTy) {
        match ty {
            Ty::Int => *index_mut(&mut self.ints, &dst.into()) = pop(&mut self.ints),
            Ty::Float => *index_mut(&mut self.floats, &dst.into()) = pop(&mut self.floats),
            Ty::Str => *index_mut(&mut self.strs, &dst.into()) = pop(&mut self.strs),
            Ty::MapIntInt => {
                *index_mut(&mut self.maps_int_int, &dst.into()) = pop(&mut self.maps_int_int)
            }
            Ty::MapIntFloat => {
                *index_mut(&mut self.maps_int_float, &dst.into()) = pop(&mut self.maps_int_float)
            }
            Ty::MapIntStr => {
                *index_mut(&mut self.maps_int_str, &dst.into()) = pop(&mut self.maps_int_str)
            }
            Ty::MapStrInt => {
                *index_mut(&mut self.maps_str_int, &dst.into()) = pop(&mut self.maps_str_int)
            }
            Ty::MapStrFloat => {
                *index_mut(&mut self.maps_str_float, &dst.into()) = pop(&mut self.maps_str_float)
            }
            Ty::MapStrStr => {
                *index_mut(&mut self.maps_str_str, &dst.into()) = pop(&mut self.maps_str_str)
            }
            Ty::Null | Ty::IterInt | Ty::IterStr => {
                panic!("unsupported register type for pop operation: {:?}", ty)
            }
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

pub(crate) fn push<T: Clone>(s: &mut Storage<T>, reg: &Reg<T>) {
    let v = index(s, reg).clone();
    s.stack.push(v);
}

pub(crate) fn pop<T: Clone>(s: &mut Storage<T>) -> T {
    s.stack.pop().expect("pop must be called on nonempty stack")
}

// Used in benchmarking code.

#[cfg(test)]
impl<T: Default> Storage<T> {
    #[cfg(feature = "unstable")]
    fn reset(&mut self) {
        self.stack.clear();
        for i in self.regs.iter_mut() {
            *i = Default::default();
        }
    }
}

#[cfg(test)]
impl<'a, LR: LineReader> Interp<'a, LR> {
    #[cfg(feature = "unstable")]
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
