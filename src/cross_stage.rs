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
    Ok(Some(match ty {
        Int => LL::LoadSlotInt(reg.into(), slot as _),
        Float => LL::LoadSlotFloat(reg.into(), slot as _),
        Str => LL::LoadSlotStr(reg.into(), slot as _),
        MapIntInt => LL::LoadSlotIntInt(reg.into(), slot as _),
        MapIntFloat => LL::LoadSlotIntFloat(reg.into(), slot as _),
        MapIntStr => LL::LoadSlotIntStr(reg.into(), slot as _),
        MapStrInt => LL::LoadSlotStrInt(reg.into(), slot as _),
        MapStrFloat => LL::LoadSlotStrFloat(reg.into(), slot as _),
        MapStrStr => LL::LoadSlotStrStr(reg.into(), slot as _),
        Null => return Ok(None),
        IterInt | IterStr => return err!("unexpected slot type: {:?}", ty),
    }))
}

pub(crate) fn store_slot_instr<'a>(reg: NumTy, ty: Ty, slot: usize) -> Result<Option<LL<'a>>> {
    use Ty::*;
    Ok(Some(match ty {
        Int => LL::StoreSlotInt(reg.into(), slot as _),
        Float => LL::StoreSlotFloat(reg.into(), slot as _),
        Str => LL::StoreSlotStr(reg.into(), slot as _),
        MapIntInt => LL::StoreSlotIntInt(reg.into(), slot as _),
        MapIntFloat => LL::StoreSlotIntFloat(reg.into(), slot as _),
        MapIntStr => LL::StoreSlotIntStr(reg.into(), slot as _),
        MapStrInt => LL::StoreSlotStrInt(reg.into(), slot as _),
        MapStrFloat => LL::StoreSlotStrFloat(reg.into(), slot as _),
        MapStrStr => LL::StoreSlotStrStr(reg.into(), slot as _),
        Null => return Ok(None),
        IterInt | IterStr => return err!("unexpected slot type: {:?}", ty),
    }))
}

fn compute_par(
    begin_refs: &HashSet<(NumTy, Ty)>,
    loop_refs: &HashSet<(NumTy, Ty)>,
    end_refs: &HashSet<(NumTy, Ty)>,
) -> SlotOps {
    let res = SlotOps {
        begin_stores: begin_refs.intersection(loop_refs).cloned().collect(),
        loop_stores: loop_refs.intersection(end_refs).cloned().collect(),
    };
    res
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
