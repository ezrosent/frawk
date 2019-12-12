use crate::builtins;
use crate::common::{self, NodeIx, NumTy, Result};
use hashbrown::HashSet;

type SmallVec<T> = smallvec::SmallVec<[T; 2]>;

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub(crate) enum BaseTy {
    // TODO(ezr): think about if we need this; I think we do because of printing.
    Null,
    Int,
    Float,
    Str,
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub(crate) enum TVar<T> {
    Iter(T),
    Scalar(T),
    Map { key: T, val: T },
}

impl<T> TVar<T> {
    // deriving Functor ...
    fn map<S>(&self, mut f: impl FnMut(&T) -> S) -> TVar<S> {
        use TVar::*;
        match self {
            Iter(t) => Iter(f(t)),
            Scalar(t) => Scalar(f(t)),
            Map { key, val } => Map {
                key: f(key),
                val: f(val),
            },
        }
    }
}

impl<T: Clone> TVar<T> {
    fn abs(&self) -> Option<TVar<Option<T>>> {
        Some(self.map(|t| Some(t.clone())))
    }
}

#[derive(Clone, Eq, PartialEq)]
enum Constraint<T> {
    // TODO(ezr): Better names for Key/KeyIn etc.
    KeyIn(T),
    Key(T),
    ValIn(T),
    Val(T),
    Flows(T),
    // TODO(ezr): have a shared Vec and just store a slice here?
    CallBuiltin(SmallVec<NodeIx>, builtins::Function),
}

impl<T> Constraint<T> {
    fn sub<S>(&self, s: S) -> Constraint<S> {
        match self {
            Constraint::KeyIn(_) => Constraint::KeyIn(s),
            Constraint::Key(_) => Constraint::Key(s),
            Constraint::ValIn(_) => Constraint::ValIn(s),
            Constraint::Val(_) => Constraint::Val(s),
            Constraint::Flows(_) => Constraint::Flows(s),
            Constraint::CallBuiltin(args, f) => Constraint::CallBuiltin(args.clone(), f.clone()),
        }
    }
    fn is_flow(&self) -> bool {
        if let Constraint::Flows(_) = self {
            true
        } else {
            false
        }
    }
}

impl Constraint<State> {
    fn eval(&self, nw: &Network) -> Result<State> {
        match self {
            Constraint::KeyIn(None) => Ok(None),
            Constraint::KeyIn(Some(TVar::Scalar(k))) => Ok(Some(TVar::Map {
                key: k.clone(),
                val: None,
            })),
            Constraint::KeyIn(op) => err!("Non-scalar KeyIn constraint: {:?}", op),

            Constraint::Key(None) => Ok(None),
            Constraint::Key(Some(TVar::Map { key: s, .. })) => Ok(Some(TVar::Scalar(s.clone()))),
            Constraint::Key(op) => {
                err!("invalid operand for Key constraint: {:?} (must be map)", op)
            }

            Constraint::ValIn(None) => Ok(None),
            Constraint::ValIn(Some(TVar::Scalar(v))) => Ok(Some(TVar::Map {
                key: None,
                val: v.clone(),
            })),
            Constraint::ValIn(op) => err!("Non-scalar ValIn constraint: {:?}", op),

            Constraint::Val(None) => Ok(None),
            Constraint::Val(Some(TVar::Iter(s)))
            | Constraint::Val(Some(TVar::Map { val: s, .. })) => Ok(Some(TVar::Scalar(s.clone()))),
            Constraint::Val(op) => err!(
                "invalid operand for Val constraint: {:?} (must be map or iterator)",
                op
            ),

            Constraint::Flows(s) => Ok(s.clone()),
            // We need to keep "feedback" because calling it repeatedly will leak.
            Constraint::CallBuiltin(args, f) => {
                let args_state: SmallVec<State> =
                    args.iter().map(|ix| nw.read(*ix).clone()).collect();
                f.step_t2(&args_state[..])
            }
        }
    }
}

/*
 *
Assuming everything is a local variable, and ignoring termination issues.
A(..) {
    return X(
             B(a),
             B(c),
             A(w));
}
B(..) {
    return Y(A(a),
             A(c),
             B(x));

so if you have x = A(y);

First, you get a fixpoint with no params assigned.

Then, for a CG you do a DFS, mutating the graph as you go but also keeping a stack of assignments.
Basically, mirroring what you'd do when interpreting the program.
}
 */

// We distinguish Edge from Constraint because we want to add more data later.
struct Edge {
    constraint: Constraint<()>,
    // TODO(ezr): store an index here? or will we replicate the rule in the graph to allow for
    // multiple call-sites?
}

// encode deps as a single edge from function to Var?

#[derive(Copy, Clone)]
enum Rule {
    Var,
    Const(TVar<BaseTy>),
}

pub(crate) type State = Option<TVar<Option<BaseTy>>>;

impl Rule {
    // TODO(ezr): Why also have `cur`? This allows us to only hand the deps back that have changed
    // since the last update. But be careful about how this will affect UDF inference.
    // TODO(ezr): Do we need to make kind constraints bidirectional? Lets see how far we get
    // without.
    // TODO(ezr): make deps an iterator?
    fn step(&self, prev: &State, deps: &[State]) -> Result<State> {
        fn value_rule(b1: BaseTy, b2: BaseTy) -> BaseTy {
            use BaseTy::*;
            match (b1, b2) {
                (Null, _) | (_, Null) | (Str, _) | (_, Str) => Str,
                (Float, _) | (_, Float) => Float,
                (Int, Int) => Int,
            }
        }
        if let Rule::Const(tv) = self {
            return Ok(tv.abs());
        }
        let mut cur = prev.clone();
        for d in deps.iter().cloned() {
            use TVar::*;
            cur = match (cur, d) {
                (None, x) | (x, None) => x,
                (Some(x), Some(y)) => match (x, y) {
                    (Iter(x), Iter(None)) | (Iter(None), Iter(x)) => Some(Iter(x)),
                    (Iter(Some(x)), Iter(Some(y))) => {
                        if x == y {
                            cur
                        } else {
                            return err!("Incompatible iterator types: {:?} vs. {:?}", x, y);
                        }
                    }
                    (Scalar(x), Scalar(None)) | (Scalar(None), Scalar(x)) => Some(Scalar(x)),
                    (Scalar(Some(x)), Scalar(Some(y))) => Some(Scalar(Some(value_rule(x, y)))),
                    // rustfmt really blows up these next two patterns.
                    #[rustfmt::skip]
                    (Map {key: Some(k), val: Some(v)}, Map {key: None, val: None}) |
                    (Map {key: None, val: None}, Map {key: Some(k), val: Some(v)}) |
                    (Map {key: Some(k), val: None}, Map {key: None, val: Some(v)}) |
                    (Map {key: None, val: Some(v)}, Map {key: Some(k), val: None})
                    => Some(Map {
                        key: Some(k),
                        val: Some(v),
                    }),

                    #[rustfmt::skip]
                    (
                        Map {key: Some(k1), val: Some(v1)},
                        Map {key: Some(k2), val: Some(v2)},
                    ) => {
                        // TODO(ezr): While we do want to promote Null => Str, this could cause
                        // some errors during compilation, as we do not support conversions for map
                        // types.
                        //
                        // The solution is to ensure a bidirectional constraint when assigning two
                        // maps. We of course may not know until later, what the kind of a variable
                        // is! That probably means we have to mutate the graph here. Can we add a
                        // "SameAs(ix, T)" constraint variant (or just give `eval` access to source
                        // and target). That doesn't really change the problem here though. You may
                        // just want to return a flag that says "all arrows bidirectional" and then
                        // make that adjustment during `solve`.
                        use BaseTy::*;
                        let key = Some(match (k1, k2) {
                            (Int, Int) => Int,
                            (_, _) => Str,
                        });
                        let val = Some(value_rule(v1, v2));
                        Some(Map { key, val })
                    }
                    (t1, t2) => return err!("kinds do not match. {:?} vs {:?}", t1, t2),
                },
            };
        }
        Ok(cur)
    }
}

fn concrete(state: State) -> TVar<BaseTy> {
    fn concrete_scalar(o: Option<BaseTy>) -> BaseTy {
        match o {
            Some(s) => s,
            None => BaseTy::Null,
        }
    }

    {
        use TVar::*;
        match state {
            Some(Iter(i)) => Iter(concrete_scalar(i)),
            Some(Scalar(i)) => Scalar(concrete_scalar(i)),
            Some(Map { key, val }) => Map {
                key: {
                    let k = concrete_scalar(key);
                    if let BaseTy::Null = k {
                        BaseTy::Str
                    } else {
                        k
                    }
                },
                val: concrete_scalar(val),
            },
            None => Scalar(BaseTy::Null),
        }
    }
}

#[derive(Clone)]
struct Node {
    rule: Rule,
    cur_val: State,
    // TODO(ezr): put debugging information in here?
}

impl Node {
    fn new(rule: Rule) -> Node {
        Node {
            rule,
            cur_val: None,
        }
    }
}

#[derive(Default)]
struct Network {
    wl: common::WorkList<NodeIx>,
    graph: common::Graph<Node, Edge>,
    iso: HashSet<(NumTy, NumTy)>,
}

fn is_iso(set: &HashSet<(NumTy, NumTy)>, n1: NodeIx, n2: NodeIx) -> bool {
    use std::cmp::{max, min};
    let i1 = n1.index() as NumTy;
    let i2 = n2.index() as NumTy;
    set.contains(&(min(i1, i2), max(i1, i2)))
}

fn make_iso(set: &mut HashSet<(NumTy, NumTy)>, n1: NodeIx, n2: NodeIx) -> bool {
    use std::cmp::{max, min};
    let i1 = n1.index() as NumTy;
    let i2 = n2.index() as NumTy;
    set.insert((min(i1, i2), max(i1, i2)))
}

impl Network {
    fn incoming(
        &self,
        ix: NodeIx,
        mut f: impl FnMut(&Edge, &Node, NodeIx) -> Result<()>,
    ) -> Result<()> {
        use petgraph::Direction::Incoming;
        let mut cur = if let Some(c) = self.graph.first_edge(ix, Incoming) {
            c
        } else {
            return Ok(());
        };
        loop {
            let e = self.graph.edge_weight(cur).unwrap();
            let n_ix = self.graph.edge_endpoints(cur).unwrap().0;
            let n = self.graph.node_weight(n_ix).unwrap();
            f(e, n, n_ix)?;
            if let Some(next) = self.graph.next_edge(cur, Incoming) {
                cur = next;
                continue;
            }
            return Ok(());
        }
    }
    fn solve(&mut self) -> Result<()> {
        let mut dep_indices: SmallVec<NodeIx> = Default::default();
        let mut deps: SmallVec<State> = Default::default();
        while let Some(ix) = self.wl.pop() {
            deps.clear();
            dep_indices.clear();
            let Node { rule, cur_val } = self.graph.node_weight(ix).unwrap().clone();
            self.incoming(ix, |edge, node, node_ix| {
                let res = edge.constraint.sub(node.cur_val).eval(self)?;
                deps.push(res);
                if edge.constraint.is_flow() {
                    dep_indices.push(node_ix);
                }
                Ok(())
            })?;
            let next = rule.step(&cur_val, &deps[..])?;
            if next == cur_val {
                continue;
            }
            if let Some(TVar::Map { .. }) = next {
                // If we have a map node, then we need to make sure that anything that assigns to
                // it winds up with the same type. Assignments are marked as `Flows` constraints;
                // and we recorded the nodes that flow into `ix` in `dep_indices`. Now we just add
                // another arrow back, unless we have done so before. We figure that last part out
                // by using the `iso` functions, which take care to sort the indices before before
                // inserting them into the hash set so we only store them once.
                for d in dep_indices.iter().cloned() {
                    if is_iso(&self.iso, ix, d) {
                        continue;
                    }
                    self.graph.add_edge(
                        ix,
                        d,
                        Edge {
                            constraint: Constraint::Flows(()),
                        },
                    );
                    make_iso(&mut self.iso, ix, d);
                }
            }
            self.graph.node_weight_mut(ix).unwrap().cur_val = next;
            for n in self.graph.neighbors(ix) {
                self.wl.insert(n);
            }
        }
        Ok(())
    }
    fn read(&self, ix: NodeIx) -> &State {
        &self.graph.node_weight(ix).unwrap().cur_val
    }
}
