use crate::common::NumTy;
use crate::compile::Ty;
use hashbrown::HashSet;

type SlotSet = HashSet<(NumTy, Ty)>;

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

// TODO: to compute these three sets, we can run all of get_global_refs, and then in addition
// accumulate the local registers used from main functions by doing a similar loop over each of the
// nodes in question, but looking for RegStatus::Local instead.
