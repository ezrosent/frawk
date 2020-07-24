use crate::common::{NumTy, Stage};
use crate::compile::Ty;
use hashbrown::HashSet;

type SlotSet = HashSet<(NumTy, Ty)>;

#[derive(Default)]
pub(crate) struct SlotOps {
    // The values stored in the BEGIN stage and loaded in the main loop stage.
    begin_stores: SlotSet,
    // The values stored in the main loop stage
    loop_stores: SlotSet,
}

pub(crate) fn compute_par(
    begin_refs: &HashSet<(NumTy, Ty)>,
    loop_refs: &HashSet<(NumTy, Ty)>,
    end_refs: &HashSet<(NumTy, Ty)>,
) -> SlotOps {
    SlotOps {
        begin_stores: begin_refs.intersection(loop_refs).cloned().collect(),
        loop_stores: loop_refs.intersection(end_refs).cloned().collect(),
    }
}

// TODO: append/prepend relevant Slot instructions for each of these results.
// TODO: do the Main check ahead of time and then pass in begin, main, end offsets into
// compute_slots.

pub(crate) fn compute_slots(
    main_offsets: &Stage<usize>,
    // TODO: from get_global_refs
    mut global_refs: Vec<HashSet<(NumTy, Ty)>>,
    // TODO: from get_local_refs(main_offsets.iter().cloned())
    local_refs: Vec<(usize, HashSet<(NumTy, Ty)>)>,
) -> SlotOps {
    for (i, s) in local_refs.into_iter() {
        global_refs[i].extend(s.into_iter())
    }
    let empty: HashSet<(NumTy, Ty)> = Default::default();
    let get_ref = |x: &Option<usize>| x.as_ref().map(|i| &global_refs[*i]).unwrap_or(&empty);
    match main_offsets {
        Stage::Main(_) => Default::default(),
        Stage::Par {
            begin,
            main_loop,
            end,
        } => compute_par(get_ref(begin), get_ref(main_loop), get_ref(end)),
    }
}
