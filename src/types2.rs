use crate::builtins;
use crate::common::{self, NodeIx, Result};

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
    Key(T),
    Val(T),
    Flows(T),
    // TODO(ezr): have a shared Vec and just store a slice here?
    CallBuiltin(SmallVec<NodeIx>, builtins::Function),
}

impl<T> Constraint<T> {
    fn sub<S>(&self, s: S) -> Constraint<S> {
        match self {
            Constraint::Key(_) => Constraint::Key(s),
            Constraint::Val(_) => Constraint::Val(s),
            Constraint::Flows(_) => Constraint::Flows(s),
            Constraint::CallBuiltin(args, f) => Constraint::CallBuiltin(args.clone(), f.clone()),
        }
    }
}

// For this all to type-check, we'll have to have "function" as a possible State. Is that okay?
// I think we could modify State and keey TVar as is. Easier: have the function _in_ the call. Have
// edges point in from some dummy node.
impl Constraint<State> {
    fn eval(&self) -> Result<State> {
        match self {
            Constraint::Key(None) => Ok(None),
            Constraint::Key(Some(TVar::Map { key: s, .. })) => Ok(Some(TVar::Scalar(s.clone()))),
            Constraint::Key(op) => {
                err!("invalid operand for Key constraint: {:?} (must be map)", op)
            }
            Constraint::Val(None) => Ok(None),
            Constraint::Val(Some(TVar::Iter(s)))
            | Constraint::Val(Some(TVar::Map { val: s, .. })) => Ok(Some(TVar::Scalar(s.clone()))),
            Constraint::Val(op) => err!(
                "invalid operand for Val constraint: {:?} (must be map or iterator)",
                op
            ),
            Constraint::Flows(s) => Ok(s.clone()),
            // We need to keep "feedback" because calling it repeatedly will leak.
            Constraint::CallBuiltin(_args, _f) => unimplemented!(),
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
}

impl Network {
    fn solve(&mut self) {
        while let Some(wl) = self.wl.pop() {
            unimplemented!()
        }
    }
}
