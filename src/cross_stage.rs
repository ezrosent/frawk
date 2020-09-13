//! This module contains helper routines for computing "slot" instructions for parallel scripts.
//!
//! If a frawk program executes its main loop in parallel, we need some mechanism for computing
//! which variables need to be propagated between stages.
use crate::common::{NumTy, Result};
use crate::compile::{Ty, LL};
use hashbrown::HashSet;

type SlotSet = HashSet<(NumTy, Ty)>;

#[derive(Default)]
pub(crate) struct SlotOps {
    // The values stored in the BEGIN stage and loaded in the main loop stage.
    pub(crate) begin_stores: SlotSet,
    // The values stored in the main loop stage and loaded in END stage.
    pub(crate) loop_stores: SlotSet,
}

pub(crate) fn load_slot_instr<'a>(reg: NumTy, ty: Ty, slot: usize) -> Result<Option<LL<'a>>> {
    use Ty::*;
    match ty {
        Int | Float | Str | MapIntInt | MapIntFloat | MapIntStr | MapStrInt | MapStrFloat
        | MapStrStr => Ok(Some(LL::LoadSlot {
            ty,
            dst: reg,
            slot: slot as _,
        })),
        Null => Ok(None),
        IterInt | IterStr => err!("unexpected slot type: {:?}", ty),
    }
}

pub(crate) fn store_slot_instr<'a>(reg: NumTy, ty: Ty, slot: usize) -> Result<Option<LL<'a>>> {
    use Ty::*;
    match ty {
        Int | Float | Str | MapIntInt | MapIntFloat | MapIntStr | MapStrInt | MapStrFloat
        | MapStrStr => Ok(Some(LL::StoreSlot {
            ty,
            src: reg,
            slot: slot as _,
        })),
        Null => Ok(None),
        IterInt | IterStr => err!("unexpected slot type: {:?}", ty),
    }
}

fn compute_par(
    begin_refs: &HashSet<(NumTy, Ty)>,
    loop_refs: &HashSet<(NumTy, Ty)>,
    end_refs: &HashSet<(NumTy, Ty)>,
) -> SlotOps {
    let begin_loop = begin_refs.intersection(loop_refs);
    let loop_end = loop_refs.intersection(end_refs);
    let begin_end = || begin_refs.intersection(end_refs);
    SlotOps {
        begin_stores: begin_loop.chain(begin_end()).cloned().collect(),
        loop_stores: loop_end.chain(begin_end()).cloned().collect(),
    }
}

/// Called from compile::Typer::add_slots().
pub(crate) fn compute_slots(
    begin: &Option<usize>,
    main_loop: &Option<usize>,
    end: &Option<usize>,
    global_refs: Vec<HashSet<(NumTy, Ty)>>,
) -> SlotOps {
    let empty: HashSet<(NumTy, Ty)> = Default::default();
    let get_ref = |x: &Option<usize>| x.as_ref().map(|i| &global_refs[*i]).unwrap_or(&empty);
    compute_par(get_ref(begin), get_ref(main_loop), get_ref(end))
}
