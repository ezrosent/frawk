//! Type inference for programs in untyped SSA form (i.e. the output of the cfg module).
//!
//! This is one of the stranger parts of frawk. If anyone has a pointer to a language that does
//! inference in this sort of way, I'd be very interested to read it!
//!
//! # Introduction
//! One somewhat interesting aspect of frawk is that it assigns static types to its variables. It
//! does this without breaking most of AWK's semantics --- the edge cases that do break could be
//! fixed fixed dynamically if need be; but for now they are breaking changes. In order to operate
//! on static types in modules "downstream" of the cfg module, we need to infer appropriate types
//! for our variables.
//!
//! This algorithm is a bit different from the standard algorithms for inferring types. Like the
//! classic [Hindley-Milner] setting, we want an algorithm that can assign types to variables and
//! functions without any annotations from the programmer. Our job is easier than this classic
//! setting in some ways, and harder in others.
//!
//! Unlike an ML-like language, we don't have all that many types. The interpreter really only
//! handles 11 or so types, and two of those (map iterators with integer and string key types) are
//! used in quite a limited way. This means that the types we infer are altogether simpler than
//! ones that are inferred in a language like Rust, OCaml or Haskell.
//!
//! # Motivation
//! Now onto the stuff that is harder. We can start by illustrating the types we assign in some
//! cases where it's not obvious how to proceed using a standard static type system as a model.
//!
//! ## One Variable, Many Types
//! However, some of these things are also harder. AWK variables can have multiple types over the
//! course of a program. It's quite easy to write:
//!
//!     x = 1         : Int
//!     x = "hello"   : String??
//!
//! An easy way to implement this would be to interpret all variables as a big sum type, and
//! changing the dynamic type of a variable as it progresses through the program. We do not do
//! this, instead we convert the program to SSA form, where most instances of this pattern become
//!
//!     x_0 = 1       : Int
//!     x_1 = "hello" : String
//!
//! Thereby keeping each variable to a single type. For variables that we cannot move into SSA form
//! (global variables, map keys), we insert coercions. Were `x` a global variable the example would
//! become:
//!
//!     x = int-to-str(1)  : String
//!     x = "hello"        : String
//!
//! *Digression*: This is probably slower than a fully dynamic implementation, but the hypothesis when
//! implementing this was that global variables that are assigned multiple types are fairly rare.
//! If that turns out not to be the case, we could make our implementation of strings closer to the
//! sum type alluded to above.
//!
//! ## Types of Operations
//! We would like to distinguish between integers and floating-point numbers. But this can make it
//! tricky to interpret the types of even very simple expressions. For example, if we just consider
//! the statement:
//!
//!     a = b + c
//!
//! The type of `a` will be a `Float` if either `b` or `c` are floats; otherwise it will be an
//! integer. That doesn't sound too disturbing, but what if the whole program is something like
//! (supposing for the moment that `b` and `c` are global):
//!
//!     a = b + c
//!     b = a + c
//!     c = a
//!
//! This is all perfectly valid; like AWK, frawk does not require variables to be declared before
//! they are used. But it is a little strange: the types of `a`, `b` and `c` all depend on the
//! types of the other two variables. Were we to "untie the knot" by stipulating that any one of
//! them was an integer or floating point value, the other two variables would take on the same
//! type. The algorithm in this module makes a "judgment call" that we should prefer integers here.
//! More importantly, it handles this caes of recursive dependencies fairly gracefully.
//!
//! # The Basic Idea
//!
//! HM-style type inference is usually implemented using a form of unification. I'm no expert, but
//! it seems like while unification could be used for inserting coercions like in the first
//! example, it's a lot messier to get working when the types of functions' return types can depend
//! on the types of the arguments. At the very least, in my experience with languages like Haskell,
//! extensions like type families often require additional manual annotations.
//!
//! This module infers types based on a (hyper)graph, where nodes indicate our current information
//! about the types of a variable and edges specify the relationship between the types in one node
//! and another. Each node except for those representing constants starts off with no information
//! about its type, but as new information arrives at a given node, it propagates that information
//! along its outgoing edges. The most common edge type is `Flows`: for assignment statements like
//!
//!     a = b
//!
//! We might add to the graph
//!
//!     b --Flows--> a
//!
//! ## Edges are Directed
//!
//! If we learn that `b` is a map or scalar, then we learn the same about `a`. Unlike a
//! unification-based system, information does not flow bidirectionally: information about the type
//! of `b` flows into `a`, but not vice-versa. To see why, consider the following example,
//! (supposing all of these variables are global, and no SSA tricks can save us):
//!
//!     b = 1
//!     c = b + 1
//!     a = b
//!     a = a "3"
//!
//! `b` is assigned a single integer value: we'd like to give it the type `Int`. However, `a` is
//! assigned to `b` as well as to the result of a string concatenation operation. String
//! concatenation produces a string, so it would seem that we have no choice but to give it the
//! type `Str`. That's what our algorithm does. It gives `b` and `c` type `Int` and `a` type `Str`,
//! necessitating an integer to string conversion in the third assignment expression. This is the
//! same number of coercions we would need with a fully dynamic representation of all the
//! variables. If, however, we let the type information for `a` flow back into `b`, then we would
//! find that `b` would have to be a string, and `c` would have to be a floating point number!
//!
//! ## Other Kinds of Edges
//!
//! In addition to `Flows`, some edges destructure map types. The `Key` constraint behaves like
//! `Flows`, but only works for `Key`; symmetically there are also `Val` constraints on maps. When
//! assigning to map keys, there is are also `KeyIn` and `MapIn`.
//!
//! For the statement:
//!
//!     m[a] = 1
//!
//! We would have the subgraph:
//!
//!     +---ValIn---Int
//!     |
//!     v
//!     m <--KeyIn-- a
//!
//! Similarly, loading from a map creates a different constraint. For:
//!
//!     x = m[a]
//!
//! We would have
//!
//!     x <--Val-- m <--KeyIn-- a
//!
//! There are corresponding constraints for iterators (the objects emitted by foreach loops).
//!
//! ## Functions
//!
//! We mentioned earlier that functions made some of this trickier. Let's focus on addition:
//!
//!     a = b + c
//!
//! By this point in the pipeline, the `+` expression is regarded as calling a "builtin function".
//! Every builtin function provides a "best guess" on its result type given partial information.
//! The full implementation can be found in the [crate::builtins] module, but for addition it will
//! guess "integer" if it knows nothing or only knows about integer operands; if it sees a float or
//! string operand it will guess "float", and if it sees a non-scalar operand it will yield a type
//! error. Because functions can be arbitrary arity, this is materialized as a "hyper-edge" in the
//! graph from all operand nodes (`b` and `c` in this case) to the result node (`a`). Whenever any
//! operand updates, we will recompute our guess about the type of the result. There are a few more
//! subtleties here (functions can have out-params), but that's the basic idea.
//!
//! User-defined functions are a great deal more complicated. Some AWK functions are polymorphic;
//! consider the function:
//!
//!     function x(a, b) { return length(a) + b; }
//!
//! It is value to pass a string _or_ any map type to parameter `a`, and it is possible to pass any
//! scalar as parameter `b`. Depending on how you count, that could be a few dozen possible
//! function types! We monomorphize functions like this: we compute all the types `x` is called
//! with and generate a separate function for each of them. When we see a function call: we first
//! take our best guess of its argument types and check to see if we have a function corresponding
//! to those arguments. If we do not, then we allocate a new node for the return type of the
//! function and build a subgraph corresponding to it. This is a bit wasteful, as it leads to
//! duplicate functions for a single callsite; we can probably improve something on that front.
//!
//! # Solving Constraints
//!
//! Once we have this graph, can push values of type [`State`] around according to the constraints
//! along the edges. Once we reach a fixed point (i.e. our guess about the types of variables stops
//! changing), we return our answer. When the answer is under-specified, we just make a guess. For
//! statements like
//!
//! ```text
//! x = 1
//! ```
//!
//! We have the graph
//!
//! ```text
//! x <--Flows-- 1
//! ```
//!
//! It starts off as
//!
//! ```text
//! x <--Flows-- 1
//! ?            Scalar(Int)
//! ```
//!
//! And then stabilizes at
//!
//! ```text
//! x <--Flows-- 1
//! Scalar(Int)  Scalar(Int)
//! ```
//!
//! For the more complicated
//!
//! ```text
//! b = 1
//! c = b + 1
//! a = b
//! a = a "3"
//! ```
//!
//! We might have
//!
//! ```text
//! +--------
//! |       |
//! |       v
//! a <--(concat)<--3
//! ^
//! |
//! Flows
//! |
//! b <--Flows-- 1
//! |            |
//! v            |
//! +<-----------+
//! |
//! Call
//! |
//! v
//! c
//! ```
//!
//! We would start with `a`, `b` and `c` all having no information associated with them. The "no
//! information" output for `concat` is String, and for `+` is Int, so in the first full iteration
//! of the solver we would have a guess that `a` is a string, and that `b` and `c` are integers.
//! Doing another full iteration (now evaluating `+` with two integer values and `concat` with a
//! string and an integer) does not change these initial guesses; so the solver would return its
//! current solution.
//!
//! This sort of achitecture is inspired by classic [static analysis algorithms] that find the least
//! fixed points of recursive equations phrased as monotone functions on some partial order with
//! finite height, along with [Propagators]
//!
//! [Propagators]: https://dspace.mit.edu/handle/1721.1/44215
//! [static analysis algorithms]: https://cs.au.dk/~amoeller/spa/
//! [Hindley-Milner]: https://en.wikipedia.org/wiki/Hindley%E2%80%93Milner_type_system
//! [`State`]: [crate::types::State]
use crate::builtins;
use crate::cfg::{self, Function, Ident, ProgramContext};
use crate::common::{self, NodeIx, NumTy, Result};
use crate::compile;
use hashbrown::{HashMap, HashSet};

use std::ops::{Deref, DerefMut};

pub(crate) type SmallVec<T> = smallvec::SmallVec<[T; 2]>;

#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub(crate) enum BaseTy {
    Null,
    Int,
    Float,
    Str,
}

#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
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
    pub(crate) fn abs(&self) -> Option<TVar<Option<T>>> {
        Some(self.map(|t| Some(t.clone())))
    }
}

#[derive(Clone, Eq, PartialEq)]
pub(crate) enum Constraint<T> {
    // TODO(ezr): Better names for Key/KeyIn etc.
    KeyIn(T),
    Key(T),
    ValIn(T),
    Val(T),
    IterVal(T),
    IterValIn(T),
    Flows(T),
    // TODO(ezr): have a shared Vec and just store a slice here?
    CallBuiltin(SmallVec<NodeIx>, builtins::Function),
    CallUDF(
        NodeIx,           /* udf node */
        SmallVec<NodeIx>, /* arg nodes */
        NumTy,            /* cfg-level function ID */
    ),
}

impl<T> Constraint<T> {
    fn sub<S>(&self, s: S) -> Constraint<S> {
        match self {
            Constraint::KeyIn(_) => Constraint::KeyIn(s),
            Constraint::Key(_) => Constraint::Key(s),
            Constraint::ValIn(_) => Constraint::ValIn(s),
            Constraint::Val(_) => Constraint::Val(s),
            Constraint::IterValIn(_) => Constraint::IterValIn(s),
            Constraint::IterVal(_) => Constraint::IterVal(s),
            Constraint::Flows(_) => Constraint::Flows(s),
            Constraint::CallBuiltin(args, f) => Constraint::CallBuiltin(args.clone(), f.clone()),
            Constraint::CallUDF(nix, args, f) => {
                Constraint::CallUDF(nix.clone(), args.clone(), f.clone())
            }
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
    fn eval<'a, 'b>(&self, tc: &mut TypeContext<'a, 'b>) -> Result<State> {
        match self {
            Constraint::KeyIn(None) => Ok(Some(TVar::Map {
                key: None,
                val: None,
            })),
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

            Constraint::ValIn(None) => Ok(Some(TVar::Map {
                key: None,
                val: None,
            })),
            Constraint::ValIn(Some(TVar::Scalar(v))) => Ok(Some(TVar::Map {
                key: None,
                val: v.clone(),
            })),
            Constraint::ValIn(op) => err!("Non-scalar ValIn constraint: {:?}", op),

            Constraint::Val(None) => Ok(None),
            Constraint::Val(Some(TVar::Map { val: s, .. })) => Ok(Some(TVar::Scalar(s.clone()))),
            Constraint::Val(op) => {
                err!("invalid operand for Val constraint: {:?} (must be map)", op)
            }

            Constraint::IterValIn(None) => Ok(Some(TVar::Iter(None))),
            Constraint::IterValIn(Some(TVar::Scalar(v))) => Ok(Some(TVar::Iter(v.clone()))),
            Constraint::IterValIn(op) => err!("Non-scalar IterValIn constraint: {:?}", op),

            Constraint::IterVal(None) => Ok(None),
            Constraint::IterVal(Some(TVar::Iter(s))) => Ok(Some(TVar::Scalar(s.clone()))),
            Constraint::IterVal(op) => err!(
                "invalid operand for IterVal constraint: {:?} (must be iterator)",
                op
            ),

            Constraint::Flows(s) => Ok(s.clone()),
            Constraint::CallBuiltin(args, f) => {
                let arg_state: SmallVec<State> =
                    args.iter().map(|ix| tc.nw.read(*ix).clone()).collect();
                f.step(&arg_state[..])
            }
            Constraint::CallUDF(nix, args, f) => {
                let ret_ix = tc.get_function(&tc.func_table[*f as usize], args.clone(), *nix);
                Ok(tc.nw.read(ret_ix).clone())
            }
        }
    }
}

// We distinguish Edge from Constraint because we want to add more data later.
#[derive(Clone)]
struct Edge {
    // TODO: do we need a separate edge type, or can it just be Constraint<()>
    constraint: Constraint<()>,
}

// encode deps as a single edge from function to Var?

#[derive(Copy, Clone)]
enum Rule {
    Var,
    Const(State),
    AlwaysNotify,
}

pub(crate) type State = Option<TVar<Option<BaseTy>>>;

pub(crate) fn null_ty() -> compile::Ty {
    flatten(concrete(None)).unwrap()
}

impl Rule {
    // TODO(ezr): Why also have `prev`? This allows us to only hand the deps back that have changed
    // since the last update. this extra functionality is not yet implemented; it's unclear if we
    // can use it while still relying on petgraph, or if we'll have to (e.g.) store a priority
    // queue of edges per node ordered by modified timestamp.
    fn step(&self, prev: &State, deps: &[State]) -> Result<(bool, State)> {
        fn value_rule(b1: BaseTy, b2: BaseTy) -> BaseTy {
            use BaseTy::*;
            match (b1, b2) {
                (Null, x) | (x, Null) => x,
                (Str, _) | (_, Str) => Str,
                (Float, _) | (_, Float) => Float,
                (Int, Int) => Int,
            }
        }
        if let Rule::Const(tv) = self {
            return Ok((tv != prev, tv.clone()));
        }
        if let Rule::AlwaysNotify = self {
            return Ok((true, None));
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
                    (Map { key: k1, val: v1 }, Map { key: k2, val: v2 }) => {
                        fn join_key(b1: BaseTy, b2: BaseTy) -> BaseTy {
                            use BaseTy::*;
                            match (b1, b2) {
                                (Float, _)
                                | (_, Float)
                                | (Str, _)
                                | (_, Str)
                                | (Null, _)
                                | (_, Null) => Str,
                                (Int, _) => Int,
                            }
                        }
                        fn lift(
                            f: impl Fn(BaseTy, BaseTy) -> BaseTy,
                            o1: Option<BaseTy>,
                            o2: Option<BaseTy>,
                        ) -> Option<BaseTy> {
                            match (o1, o2) {
                                (Some(x), Some(y)) => Some(f(x, y)),
                                (Some(x), None) | (None, Some(x)) => Some(x),
                                (None, None) => None,
                            }
                        }
                        Some(Map {
                            key: lift(join_key, k1, k2),
                            val: lift(value_rule, v1, v2),
                        })
                    }
                    (t1, t2) => return err!("kinds do not match. {:?} vs {:?}", t1, t2),
                },
            };
        }
        Ok((prev != &cur, cur))
    }
}

fn concrete(state: State) -> TVar<BaseTy> {
    fn concrete_scalar(o: &Option<BaseTy>) -> BaseTy {
        o.unwrap_or(BaseTy::Null)
    }
    match state {
        Some(x) => x.map(concrete_scalar),
        None => TVar::Scalar(BaseTy::Null),
    }
}

fn flatten(tv: TVar<BaseTy>) -> Result<compile::Ty> {
    use compile::Ty;
    use {BaseTy::*, TVar::*};
    fn flatten_base(b: BaseTy) -> Ty {
        match b {
            Int => Ty::Int,
            Float => Ty::Float,
            Str => Ty::Str,
            Null => Ty::Null,
        }
    }
    match tv {
        Scalar(b) => Ok(flatten_base(b)),
        Iter(Int) => Ok(Ty::IterInt),
        Iter(Null) | Iter(Str) => Ok(Ty::IterStr),
        Iter(x) => err!("Iterator over an unsupported type: {:?}", x),
        Map { key, val } => {
            let f = |ty| {
                if ty == Null {
                    Ty::Str
                } else {
                    flatten_base(ty)
                }
            };
            match (f(key), f(val)) {
                (Ty::Int, Ty::Int) => Ok(Ty::MapIntInt),
                (Ty::Int, Ty::Float) => Ok(Ty::MapIntFloat),
                (Ty::Int, Ty::Str) => Ok(Ty::MapIntStr),
                (Ty::Str, Ty::Int) => Ok(Ty::MapStrInt),
                (Ty::Str, Ty::Float) => Ok(Ty::MapStrFloat),
                (Ty::Str, Ty::Str) => Ok(Ty::MapStrStr),
                (k, v) => err!("Map with unsupported type (key={:?} val={:?})", k, v),
            }
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

pub(crate) struct Network {
    base_node: NodeIx,
    wl: common::WorkList<NodeIx>,
    call_deps: HashMap<NodeIx, SmallVec<NodeIx>>,
    graph: common::Graph<Node, Edge>,
    iso: HashSet<(NumTy, NumTy)>,
}

impl Default for Network {
    fn default() -> Network {
        let mut graph = common::Graph::default();
        let base_node = graph.add_node(Node::new(Rule::Var));
        Network {
            graph,
            base_node,
            wl: Default::default(),
            call_deps: Default::default(),
            iso: Default::default(),
        }
    }
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
    fn add_rule(&mut self, rule: Rule) -> NodeIx {
        let res = self.graph.add_node(Node::new(rule));
        self.wl.insert(res);
        res
    }

    fn read(&self, ix: NodeIx) -> &State {
        &self.graph.node_weight(ix).unwrap().cur_val
    }

    pub(crate) fn add_dep(&mut self, from: NodeIx, to: NodeIx, constraint: Constraint<()>) {
        self.graph.add_edge(from, to, Edge { constraint });
        self.wl.insert(from);
    }
}

pub(crate) struct TypeContext<'a, 'b> {
    pub(crate) nw: Network,
    base: HashMap<State, NodeIx>,
    env: HashMap<Args<Ident>, NodeIx>,
    funcs: HashMap<Args<NumTy>, NodeIx>,
    maps: HashSet<NodeIx>,
    func_table: &'a [Function<'b, &'b str>],
    local_globals: &'a HashSet<NumTy>,
    udf_nodes: Vec<NodeIx>,
}

struct View<'a, 'b, 'c> {
    tc: &'a mut TypeContext<'b, 'c>,
    frame_id: NumTy, // which function are we in?
    frame_args: SmallVec<State>,
}

impl<'a, 'b, 'c> Deref for View<'a, 'b, 'c> {
    type Target = &'a mut TypeContext<'b, 'c>;
    fn deref(&self) -> &&'a mut TypeContext<'b, 'c> {
        &self.tc
    }
}

impl<'a, 'b, 'c> DerefMut for View<'a, 'b, 'c> {
    fn deref_mut(&mut self) -> &mut &'a mut TypeContext<'b, 'c> {
        &mut self.tc
    }
}

pub(crate) fn get_types<'a>(pc: &ProgramContext<'a, &'a str>) -> Result<TypeInfo> {
    TypeContext::from_prog(pc)
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Args<T> {
    id: T,
    func_id: Option<NumTy>,
    args: SmallVec<State>, // ignored when id.global
}

pub(crate) struct TypeInfo {
    // Map a particular identifier in a function to a type.
    pub var_tys: HashMap<(Ident, NumTy, SmallVec<compile::Ty>), compile::Ty>,
    // Map a particular function invocation to a return type.
    pub func_tys: HashMap<(NumTy, SmallVec<compile::Ty>), compile::Ty>,
}

impl<'b, 'c> TypeContext<'b, 'c> {
    fn from_pc(pc: &'b ProgramContext<'c, &'c str>) -> TypeContext<'b, 'c> {
        let mut tc = TypeContext {
            nw: Default::default(),
            base: Default::default(),
            env: Default::default(),
            funcs: Default::default(),
            maps: Default::default(),
            func_table: &pc.funcs[..],
            local_globals: pc.local_globals_ref(),
            udf_nodes: Default::default(),
        };
        tc.udf_nodes = (0..pc.funcs.len())
            .map(|_| tc.nw.add_rule(Rule::AlwaysNotify))
            .collect();
        tc
    }
    pub(crate) fn from_prog<'a>(pc: &ProgramContext<'a, &'a str>) -> Result<TypeInfo> {
        use hashbrown::hash_map::Entry;
        let mut tc = TypeContext::from_pc(pc);
        // TODO: to migrate, simply iterate over the Stage variant of this and solve at the end?
        for offset in pc.main_offsets() {
            let main = &pc.funcs[offset];
            let main_base = tc.udf_nodes[offset];
            tc.get_function(main, /*arg_nodes=*/ Default::default(), main_base);
        }
        tc.solve()?;
        let mut var_tys = HashMap::new();
        let mut func_tys = HashMap::new();
        for (Args { id, args, .. }, ix) in tc.funcs.iter() {
            let mut flat_args = SmallVec::new();
            for a in args.iter().cloned() {
                flat_args.push(flatten(concrete(a))?);
            }
            if let Entry::Vacant(v) = func_tys.entry((*id, flat_args)) {
                v.insert(flatten(concrete(*tc.nw.read(*ix)))?);
            }
        }
        for (Args { id, func_id, args }, ix) in tc.env.iter() {
            let mut flat_args = SmallVec::new();
            for a in args.iter().cloned() {
                flat_args.push(flatten(concrete(a))?);
            }
            let v = flatten(concrete(*tc.nw.read(*ix)))?;

            // We won't use the function id if id.global, so setting it to 0 should be fine.
            // TODO clean up some of this to make it less misleading
            match var_tys.entry((*id, func_id.unwrap_or(0), flat_args)) {
                Entry::Vacant(vac) => {
                    vac.insert(v);
                }
                Entry::Occupied(mut occ) => {
                    // In an earlier iteration wrt args, a variable could have been assigned Int,
                    // but is now assigned Float. Pick the float copy, otherwise signal an error.
                    //
                    // TODO: this is a bit of a hack, This is only necessary because we aren't
                    // filtering out calls to functions with (None) arguments, after solving. In
                    // many cases, those functions are not here anymore. We should key this on
                    // unflattened args.
                    let prev = *occ.get();
                    if prev != v {
                        use compile::Ty;
                        match (prev, v) {
                            // TODO: unclear if this is safe in all cases of Str => Float. It only
                            // makes sense when the Str _was_ null. Perhaps we can propagate that
                            // null information into this map.
                            (Ty::Float, _) => {}
                            (_, Ty::Float) => {
                                occ.insert(v);
                            }
                            _ => {
                                return err!(
                                    "coherence violation! (func_id={:?}) {:?} in args {:?}, we get both {:?} and {:?}\nenv={:?}",
                                    func_id,
                                    id,
                                    args,
                                    v,
                                    prev,
                                    tc.env
                                );
                            }
                        }
                    }
                }
            }
        }
        Ok(TypeInfo { var_tys, func_tys })
    }
    fn solve(&mut self) -> Result<()> {
        let mut dep_indices: SmallVec<NodeIx> = Default::default();
        let mut deps: SmallVec<State> = Default::default();
        // TODO are updates to UDFs getting "lost" somehow? It seems like it isn't just changes in
        // arguments that could cause a change in the return type (imagine returning a global
        // variable from within a function). How do we ensure we see those updates?
        //  - Each call edge (or each UDF?) has its own node. Changes to return type cause that node to be
        //  inserted into the worklist? Downside is that all callsites are woken up; but we may not
        //  have a choice?
        while let Some(ix) = self.nw.wl.pop() {
            deps.clear();
            dep_indices.clear();
            let Node { rule, cur_val } = self.nw.graph.node_weight(ix).unwrap().clone();
            // Iterate over the incoming edges; read their current values and evaluate the
            // constraints.
            use petgraph::Direction::Incoming;
            let mut walker = self.nw.graph.neighbors_directed(ix, Incoming).detach();
            while let Some((e_ix, node_ix)) = walker.next(&self.nw.graph) {
                let edge = self.nw.graph.edge_weight(e_ix).unwrap().clone();
                let node_val = self.nw.graph.node_weight(node_ix).unwrap().cur_val.clone();
                deps.push(edge.constraint.sub(node_val).eval(self)?);
                if edge.constraint.is_flow() {
                    dep_indices.push(node_ix);
                }
            }
            // Compute an update value based on the newly-evaluated constraints.
            let (changed, next) = rule.step(&cur_val, &deps[..])?;
            if !changed {
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
                    if is_iso(&self.nw.iso, ix, d) {
                        continue;
                    }
                    self.nw.graph.add_edge(
                        ix,
                        d,
                        Edge {
                            constraint: Constraint::Flows(()),
                        },
                    );
                    make_iso(&mut self.nw.iso, ix, d);
                }
            }
            self.nw.graph.node_weight_mut(ix).unwrap().cur_val = next;
            for n in self.nw.graph.neighbors(ix) {
                self.nw.wl.insert(n);
            }
            for c in self.nw.call_deps.get(&ix).iter().flat_map(|c| c.iter()) {
                self.nw.wl.insert(*c);
            }
        }
        Ok(())
    }

    pub(crate) fn constant(&mut self, tv: State) -> NodeIx {
        use hashbrown::hash_map::Entry::*;
        match self.base.entry(tv) {
            Occupied(o) => *o.get(),
            Vacant(v) => {
                let res = self.nw.add_rule(Rule::Const(tv));
                v.insert(res);
                res
            }
        }
    }
    pub(crate) fn constrain_as_map(&mut self, ix: NodeIx) {
        // To be completely explicit, this function assigns a unique `Flows` constaint into a map
        // from the constant node that "just specifies the node is a Map".
        if self.maps.insert(ix) {
            let is_map = self.constant(Some(TVar::Map {
                key: None,
                val: None,
            }));
            self.nw.add_dep(is_map, ix, Constraint::Flows(()))
        }
    }
    pub(crate) fn get_node(&mut self, key: Args<Ident>) -> NodeIx {
        self.env
            .entry(key)
            .or_insert(self.nw.add_rule(Rule::Var))
            .clone()
    }

    fn get_function<'a>(
        &mut self,
        Function {
            ident, cfg, args, ..
        }: &Function<'a, &'a str>,
        mut arg_nodes: SmallVec<NodeIx>,
        base_node: NodeIx,
    ) -> NodeIx {
        // First we want to normalize the provided arguments. If we provide too few arguments, the
        // rest are filled with nulls. If we provide too many arguments, we throw away the extras.
        if arg_nodes.len() < args.len() {
            for _ in 0..(args.len() - arg_nodes.len()) {
                arg_nodes.push(self.constant(None));
            }
        }
        if args.len() < arg_nodes.len() {
            arg_nodes.truncate(args.len());
        }

        let arg_states = arg_nodes
            .iter()
            .map(|ix| self.nw.read(*ix).clone())
            .collect();

        let key = Args {
            id: *ident,
            func_id: None,
            args: arg_states,
        };

        // Check if we have already created the function
        if let Some(ix) = self.funcs.get(&key) {
            return *ix;
        }
        // Just return a new node for the return value. We will infer return types by looking up
        // this node later and adding dependencies when we encounter a `Return` stmt.
        //
        // TODO: this means we do some duplicate work in rewriting returns in the cfg module.

        // Create a new function.
        let res = self.nw.add_rule(Rule::Var);
        self.nw.add_dep(res, base_node, Constraint::Flows(()));
        self.funcs.insert(key.clone(), res);
        let mut view = View {
            tc: self,
            frame_id: *ident,
            frame_args: key.args.clone(),
        };

        // Apply the arguments appropriately:
        for (cfg::Arg { id, .. }, arg_node) in args.iter().zip(arg_nodes.iter().cloned()) {
            let ix = view.ident_node(id);
            view.nw.add_dep(arg_node, ix, Constraint::Flows(()));
        }
        let nodes = cfg.raw_nodes();
        for bb in nodes {
            for stmt in bb.weight.q.iter() {
                view.constrain_stmt(stmt);
            }
        }
        res
    }
}

impl<'b, 'c, 'd> View<'b, 'c, 'd> {
    fn add_builtin_call(&mut self, f: builtins::Function, args: SmallVec<NodeIx>, to: NodeIx) {
        f.feedback(&args[..], self);
        for arg in args.iter() {
            self.nw
                .call_deps
                .entry(*arg)
                .or_insert(Default::default())
                .push(to);
        }
        let from = self.nw.base_node;
        self.nw.add_dep(from, to, Constraint::CallBuiltin(args, f));
        self.nw.wl.insert(to);
    }
    fn add_udf_call(&mut self, f: NumTy, args: SmallVec<NodeIx>, to: NodeIx) {
        for arg in args.iter() {
            self.nw
                .call_deps
                .entry(*arg)
                .or_insert(Default::default())
                .push(to);
        }
        let from = self.udf_nodes[f as usize];
        self.nw
            .add_dep(from, to, Constraint::CallUDF(from, args, f));
        self.nw.wl.insert(to);
    }

    fn constrain_stmt<'a>(&mut self, stmt: &cfg::PrimStmt<'a>) {
        use cfg::PrimStmt::*;
        match stmt {
            AsgnIndex(arr, ix, v) => {
                let arr_ix = self.ident_node(arr);
                let ix_ix = self.val_node(ix);
                self.constrain_as_map(arr_ix);
                // TODO(ezr): set up caching for keys, values of maps and iterators?
                self.nw.add_dep(ix_ix, arr_ix, Constraint::KeyIn(()));
                let val_ix = self.nw.add_rule(Rule::Var);
                self.nw.add_dep(val_ix, arr_ix, Constraint::ValIn(()));
                self.constrain_expr(v, val_ix);
            }
            AsgnVar(v, e) => {
                let v_ix = self.ident_node(v);
                self.constrain_expr(e, v_ix);
            }
            Return(v) => {
                let v_ix = self.val_node(v);
                let cur_func_key = Args {
                    id: self.frame_id,
                    func_id: None,
                    args: self.frame_args.clone(),
                };
                let ret_ix = self.tc.funcs[&cur_func_key];
                self.nw.add_dep(v_ix, ret_ix, Constraint::Flows(()));
            }
            Printf(fmt, args, out) => {
                // Printf's arguments can be any scalar.
                //
                // NB: This information isn't really needed for inference, but it means that we can
                // avoid extra type-checking when lowering to typed bytecode. It's worth revisiting
                // this form a performance standpoint.
                let is_scalar: State = Some(TVar::Scalar(None));
                let scalar_node = self.constant(is_scalar);
                let fmt_node = self.val_node(fmt);
                self.nw
                    .add_dep(scalar_node, fmt_node, Constraint::Flows(()));
                for a in args.iter() {
                    let arg_node = self.val_node(a);
                    self.nw
                        .add_dep(scalar_node, arg_node, Constraint::Flows(()));
                }
                if let Some((out, _append)) = out {
                    let out_node = self.val_node(out);
                    self.nw
                        .add_dep(scalar_node, out_node, Constraint::Flows(()));
                }
            }
            // Builtins have fixed types; no constraint generation is necessary.
            // For IterDrop, we do not add extra constraints because IterBegin and IterNext will be
            // sufficient to determine the type of a given iterator.
            IterDrop(_) | SetBuiltin(_, _) => {}
            // Attempting something different for PrintAll vs Printf; the constraints are similar,
            // but looking at deferring the type checks to the `compile` phase.
            PrintAll(..) => {}
        }
    }

    fn constrain_expr<'a>(&mut self, expr: &cfg::PrimExpr<'a>, to: NodeIx) {
        use cfg::PrimExpr::*;
        match expr {
            Val(pv) => {
                let pv_ix = self.val_node(pv);
                self.nw.add_dep(pv_ix, to, Constraint::Flows(()));
            }
            Phi(preds) => {
                for (_, id) in preds.iter() {
                    let id_ix = self.ident_node(id);
                    self.nw.add_dep(id_ix, to, Constraint::Flows(()));
                }
            }
            CallBuiltin(f, args) => {
                let args: SmallVec<NodeIx> = args.iter().map(|arg| self.val_node(arg)).collect();
                self.add_builtin_call(*f, args, to);
            }
            CallUDF(f, args) => {
                let args: SmallVec<NodeIx> = args.iter().map(|arg| self.val_node(arg)).collect();
                self.add_udf_call(*f, args, to);
            }
            Sprintf(fmt, args) => {
                let str_node = self.constant(TVar::Scalar(BaseTy::Str).abs());
                let is_scalar: State = Some(TVar::Scalar(None));
                let scalar_node = self.constant(is_scalar);
                for a in args {
                    let a_node = self.val_node(a);
                    self.nw.add_dep(scalar_node, a_node, Constraint::Flows(()));
                }
                let fmt_node = self.val_node(fmt);
                self.nw
                    .add_dep(scalar_node, fmt_node, Constraint::Flows(()));
                self.nw.add_dep(str_node, to, Constraint::Flows(()));
            }
            Index(arr, ix) => {
                let arr_ix = self.val_node(arr);
                let ix_ix = self.val_node(ix);
                self.constrain_as_map(arr_ix);
                self.nw.add_dep(arr_ix, ix_ix, Constraint::Key(()));
                self.nw.add_dep(arr_ix, to, Constraint::Val(()));
            }
            IterBegin(arr) => {
                let arr_ix = self.val_node(arr);
                self.constrain_as_map(arr_ix);
                let iter_ix = to;
                let key_ix = self.nw.add_rule(Rule::Var);

                // The `key_ix` is a proxy node that has a bidirectional constraint with the key of
                // the array and the value of the iterator. Its presence ensures that the
                // iterator's type and the map's key type remain the same.
                self.nw.add_dep(key_ix, arr_ix, Constraint::KeyIn(()));
                self.nw.add_dep(arr_ix, key_ix, Constraint::Key(()));
                self.nw.add_dep(key_ix, iter_ix, Constraint::IterValIn(()));
                self.nw.add_dep(iter_ix, key_ix, Constraint::IterVal(()));
            }
            HasNext(_) => {
                let int = self.constant(TVar::Scalar(BaseTy::Int).abs());
                self.nw.add_dep(int, to, Constraint::Flows(()))
            }
            Next(iter) => {
                let iter_ix = self.val_node(iter);
                self.nw.add_dep(iter_ix, to, Constraint::IterVal(()));
            }
            LoadBuiltin(bv) => {
                let bv_ix = self.constant(bv.ty().abs());
                self.nw.add_dep(bv_ix, to, Constraint::Flows(()));
            }
        };
    }

    fn val_node<'a>(&mut self, val: &cfg::PrimVal<'a>) -> NodeIx {
        use cfg::PrimVal::*;
        match val {
            Var(id) => self.ident_node(id),
            ILit(_) => self.constant(TVar::Scalar(BaseTy::Int).abs()),
            FLit(_) => self.constant(TVar::Scalar(BaseTy::Float).abs()),
            StrLit(_) => self.constant(TVar::Scalar(BaseTy::Str).abs()),
        }
    }

    fn is_global(&self, id: &Ident) -> bool {
        id.is_global(&self.local_globals)
    }

    fn ident_node(&mut self, id: &Ident) -> NodeIx {
        let is_global = self.is_global(id);
        let key = Args {
            id: *id,
            func_id: if is_global { None } else { Some(self.frame_id) },
            args: if is_global {
                Default::default()
            } else {
                self.frame_args.clone()
            },
        };
        self.get_node(key)
    }
}
