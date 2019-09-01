//! Algorithms and types pertaining to type deduction and converion.
//!
//! TODO: update this with more documentation when the algorithms are more fully baked.
use crate::common::{CompileError, Graph, NodeIx, Result};
use std::hash::Hash;
type SmallVec<T> = crate::smallvec::SmallVec<[T; 2]>;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(crate) enum Scalar {
    Str,
    Int,
    Float,
}

pub(crate) struct MapTy {
    key: Option<Scalar>,
    val: Option<Scalar>,
}

/// A propagator is a monotone function that receives "partially done" inputs and produces
/// "partially done" outputs. This is a very restricted API to simplify the implementation of a
/// propagator network.
///
/// TODO: Link to more information on propagators.
pub(crate) trait Propagator: Default + Clone {
    type Item;
    /// While rules (instances of Self) themselves will propgate information in a monotone fashion,
    /// we may run into cases where we want to change from one rule to another. This too has to be
    /// "monotone" in the same way.
    ///
    /// Using TypeRule as an example, we cannot change a variable from an Int to a Str. But we might
    /// be able to refine a Placeholder into either one of those. Updating to an operation with
    /// strictly less information (e.g. ArithOp(..) => Placeholder) has no effect; in this way,
    /// try_replace behaves like a lub operation.
    ///
    /// N.B In a more general propagator implementation, nodes in a network could have different;
    /// and then rules could just be nodes whose values were functions. It may be worth implementing
    /// such an abstraction depending on how this implementation grows in complexity.
    fn try_replace(&mut self, other: Self) -> Option<Self>;
    /// Ingest new information from `incoming`, produce a new output and indicate if the output has
    /// been saturated (i.e. regardless of what new inputs it will produce, the output will not
    /// change).
    fn step(&mut self, incoming: impl Iterator<Item = Self::Item>)
        -> (bool /* done */, Self::Item);
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) enum TypeRule {
    Placeholder,
    // For literals, and also operators like '/' (which will always coerce to float) or bitwise
    // operations (which always coerce to integers).
    Const(Scalar),
    // For things like +,% and *
    ArithOp(Option<Scalar>),
    CompareOp(Option<Scalar>),
    MapKey(Option<Scalar>),
    MapVal(Option<Scalar>),
}

impl Default for TypeRule {
    fn default() -> TypeRule {
        TypeRule::Placeholder
    }
}

fn apply_binary_rule<T>(
    mut start: T,
    incoming: impl Iterator<Item = T>,
    f: impl Fn(&T, &T) -> (bool, T),
) -> (bool, T) {
    let mut done = false;
    for i in incoming {
        let (stop, cur) = f(&start, &i);
        start = cur;
        done = stop;
        if stop {
            break;
        }
    }
    (done, start)
}

impl Propagator for TypeRule {
    type Item = Option<Scalar>;
    fn try_replace(&mut self, other: TypeRule) -> Option<TypeRule> {
        use TypeRule::*;
        if let Placeholder = self {
            *self = other;
            return None;
        }
        match (self, other) {
            (Placeholder, _) => unreachable!(),
            (ArithOp(Some(_)), ArithOp(None))
            | (CompareOp(Some(_)), CompareOp(None))
            | (MapKey(Some(_)), MapKey(None))
            | (MapVal(Some(_)), MapVal(None))
            | (_, Placeholder) => None,
            (_, other) => Some(other),
        }
    }
    fn step(
        &mut self,
        incoming: impl Iterator<Item = Option<Scalar>>,
    ) -> (bool /* done */, Option<Scalar>) {
        fn op_helper(o1: &Option<Scalar>, o2: &Option<Scalar>) -> (bool, Option<Scalar>) {
            use Scalar::*;
            match (o1, o2) {
                // No information to propagate
                (None, None) => (false, None),
                (Some(Str), _) | (_, Some(Str)) | (Some(Float), _) | (_, Some(Float)) => {
                    (true, Some(Float))
                }
                (Some(Int), _) | (_, Some(Int)) => (false, Some(Int)),
            }
        }
        use TypeRule::*;
        match self {
            Placeholder => (false, None),
            Const(s) => (true, Some(*s)),
            ArithOp(ty) => {
                let (done, res) = apply_binary_rule(*ty, incoming, op_helper);
                *ty = res;
                (done, res)
            }
            CompareOp(ty) => {
                let (done, res) = apply_binary_rule(*ty, incoming, op_helper);
                *ty = res;
                (done, Some(Scalar::Int))
            }
            MapKey(ty) => {
                let (done, res) = apply_binary_rule(*ty, incoming, |t1, t2| {
                    use Scalar::*;
                    match (t1, t2) {
                        (None, None) => (false, None),
                        (Some(Str), _) | (_, Some(Str)) | (Some(Float), _) | (_, Some(Float)) => {
                            (true, Some(Str))
                        }
                        (Some(Int), _) | (_, Some(Int)) => (false, Some(Int)),
                    }
                });
                *ty = res;
                (done, res)
            }
            MapVal(ty) => {
                let (done, res) = apply_binary_rule(*ty, incoming, |t1, t2| {
                    use Scalar::*;
                    match (t1, t2) {
                        (None, None) => (false, None),
                        (Some(Str), _) | (_, Some(Str)) => (true, Some(Str)),
                        (Some(Float), _) | (_, Some(Float)) => (true, Some(Float)),
                        (Some(Int), _) | (_, Some(Int)) => (false, Some(Int)),
                    }
                });
                *ty = res;
                (done, res)
            }
        }
    }
}

struct Node<P: Propagator> {
    // The propagator rule.
    prop: P,
    // The last value returned by the rule, or the default value.
    item: P::Item,
    // Will the propagator return new values?
    done: bool,
    // Is this node in the worklist?
    in_wl: bool,
}

#[derive(Default)]
pub(crate) struct Network<P: Propagator> {
    g: Graph<Node<P>, ()>,
    wl: Vec<NodeIx>,
}

impl<P: Propagator + std::fmt::Debug> Network<P>
where
    P::Item: Eq + Clone + Default,
{
    pub(crate) fn insert(&mut self, p: P, deps: impl Iterator<Item = NodeIx>) -> NodeIx {
        let ix = self.g.add_node(Node {
            prop: p,
            item: Default::default(),
            done: false,
            in_wl: true,
        });
        for d in deps {
            self.g.add_edge(d, ix, ());
        }
        self.wl.push(ix);
        ix
    }

    pub(crate) fn update(
        &mut self,
        ix: NodeIx,
        p: P,
        deps: impl Iterator<Item = NodeIx>,
    ) -> Result<()> {
        {
            let Node {
                prop,
                item: _,
                done: _,
                in_wl,
            } = self.g.node_weight_mut(ix).unwrap();
            if let Some(p) = prop.try_replace(p) {
                // Return a result here if we think this would represent a malformed program, not
                // just a bug in SSA conversion.
                return err!(
                    "internal error: tried to overwrite {:?} rule with {:?}",
                    prop,
                    p
                );
            }
            *in_wl = true;
        }
        for d in deps {
            self.g.add_edge(d, ix, ());
        }
        self.wl.push(ix);
        Ok(())
    }
    pub(crate) fn read(&self, ix: NodeIx) -> &P::Item {
        &self.g.node_weight(ix).unwrap().item
    }
    fn solve(&mut self) {
        let mut incoming: SmallVec<P::Item> = Default::default();
        let mut neighs: SmallVec<NodeIx> = Default::default();
        while let Some(node) = self.wl.pop() {
            use petgraph::Direction::*;
            let mut p = {
                let Node {
                    prop,
                    item: _,
                    done,
                    in_wl,
                } = self.g.node_weight_mut(node).unwrap();
                *in_wl = false;
                if *done {
                    continue;
                }
                prop.clone()
            };
            // TODO: Add support for something like timestamps so that we could filter for nodes
            // that have changed since the last iteration. Probably not a super high-priority
            // feature, but could be useful if we made this more general. (But for full generality
            // we could also probably just use timely/differential).
            incoming.extend(
                self.g
                    .neighbors_directed(node, Incoming)
                    .map(|ix| self.g.node_weight(ix).unwrap().item.clone()),
            );
            let (done_now, ty) = p.step(incoming.drain());
            let Node {
                prop,
                item,
                done,
                in_wl: _,
            } = self.g.node_weight_mut(node).unwrap();
            *done = done_now;
            *prop = p;
            if item != &ty {
                *item = ty;
                neighs.extend(self.g.neighbors_directed(node, Outgoing));
                for n in neighs.drain() {
                    self.wl.push(n);
                    *(&mut self.g.node_weight_mut(n).unwrap().in_wl) = true;
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn plus_key_int() {
        let mut n = Network::default();
        use Scalar::*;
        use TypeRule::*;
        let i1 = n.insert(Const(Int), None.into_iter());
        let i2 = n.insert(Placeholder, None.into_iter());
        let addi12 = n.insert(ArithOp(None), vec![i1, i2].into_iter());
        n.update(i2, MapKey(None), vec![addi12].into_iter());
        n.solve();
        assert_eq!(n.read(i1), &Some(Int));
        assert_eq!(n.read(i2), &Some(Int));
        assert_eq!(n.read(addi12), &Some(Int));
    }

    #[test]
    fn plus_key_float() {
        let mut n = Network::default();
        use Scalar::*;
        use TypeRule::*;
        let i1 = n.insert(Const(Int), None.into_iter());
        let f1 = n.insert(Placeholder, None.into_iter());
        let add12 = n.insert(ArithOp(None), vec![i1, f1].into_iter());
        let f2 = n.insert(Const(Float), None.into_iter());
        n.update(f1, MapKey(None), vec![add12, f2].into_iter());
        n.solve();
        assert_eq!(n.read(i1), &Some(Int));
        assert_eq!(n.read(f1), &Some(Str));
        assert_eq!(n.read(f2), &Some(Float));
        assert_eq!(n.read(add12), &Some(Float));
    }

}
