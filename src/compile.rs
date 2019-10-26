use crate::bytecode::{Interp,Instr};
use crate::cfg;
use crate::types::get_types;

pub(crate) fn bytecode<'a,'b>(ctx: cfg::Context<'a, &'b str>) -> (Interp<'a>, Vec<Instr<'a>>) {
    let ts = get_types(ctx.cfg(), ctx.num_idents());

    unimplemented!()
}


pub(crate) fn _test() {}
