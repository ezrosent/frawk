//! Compute dominance frontiers for a control flow graph. See the comments for `DomInfo` for more
//! information.
use crate::common::{Graph, NodeIx, NumTy};
use hashbrown::HashSet;
use petgraph::Direction;
use smallvec::SmallVec;

pub(crate) type Tree = Vec<SmallVec<[NumTy; 2]>>;
pub(crate) type Frontier = Vec<HashSet<NumTy>>;

/// Compute the [dominator tree and dominance frontier][0] for a control-flow graph. We use the
/// Semi-NCA algorithm from ["Finding Dominators in Practice"][1] by Georgiadis et. al.  to compute
/// the dominator tree, and then use the algorithm from ["A Simple, Fast Dominance Algorithm"][2]
/// by Cooper et.  al. for building dominance frontiers from the dominator tree. The ["Tiger
/// Book"][3] by Appel was a helpful reference for computing semidominators.
///
/// A brief note on why we are not using the iterative algorithm from the Cooper paper. There is a
/// strong reason to do so: the algorithm is a bit simpler to implement, and if you iterate in
/// reverse post-order as they suggest, the algorithm will complete in a single pass if the flow
/// graph is reducible. While most AWK programs do have reducible CFGs, I want to hold open the
/// possibility of adding proper tail calls to AWK functions at some point in the future, in which
/// case Semi-NCA will handle more pathological cases.
///
/// [0]: https://en.wikipedia.org/wiki/Dominator_(graph_theory)
/// [1]: http://jgaa.info/accepted/2006/GeorgiadisTarjanWerneck2006.10.1.pdf
/// [2]: https://www.cs.rice.edu/~keith/EMBED/dom.pdf
/// [3]: https://www.cs.princeton.edu/~appel/modern/
pub(crate) struct DomInfo<'a, V, E> {
    // `cfg` must be a flow graph: all nodes must be reachable from `entry`.
    cfg: &'a Graph<V, E>,
    entry: NodeIx,

    // Semi-NCA metadata, indexed by NodeIndex
    info: Vec<NodeInfo>,
    // (pre-order) depth-first ordering of nodes.
    dfs: Vec<NodeIx>,
    // Used in semidominator calculation.
    // ancestor: Vec<NumTy>,
    // TODO: should this go in NodeInfo?
    best: Vec<NumTy>,
}

impl<'a, V, E> DomInfo<'a, V, E> {
    pub fn new(g: &'a Graph<V, E>, entry: NodeIx) -> Self {
        let mut res = DomInfo {
            cfg: g,
            entry,
            info: vec![Default::default(); g.node_count()],
            dfs: Default::default(),
            best: vec![NODEINFO_UNINIT; g.node_count()],
        };
        res.dfs(res.entry, NODEINFO_UNINIT);
        res.semis();
        res.idoms();
        res
    }

    // a.k.a AncestorWithLowestSemi in Tiger Book
    fn eval(&mut self, node: impl HasNum) -> NumTy {
        let p = self.at(node).ancestor;
        debug_assert!(
            p != NODEINFO_UNINIT,
            "node n={} has uninitialized ancestor. {:?}",
            node.ix(),
            self.at(node),
        );
        if self.at(p).ancestor != NODEINFO_UNINIT {
            let b = self.eval(p);
            *(&mut self.at_mut(node).ancestor) = self.at(p).ancestor;
            if self.at(self.at(b).sdom).dfsnum < self.at(self.at(self.best_at(node)).sdom).dfsnum {
                self.set_best(node, b);
            }
        }
        self.best_at(node)
    }

    fn dfs(&mut self, cur_node: NodeIx, parent: NumTy) {
        // TODO: consider explicit maintenance of stack.
        //       probably not a huge win performance-wise, but it could avoid stack overflow on
        //       pathological inputs.
        debug_assert!(!self.at(cur_node).seen());
        let seen_so_far = self.seen();
        *self.at_mut(cur_node) = NodeInfo {
            dfsnum: seen_so_far,
            parent,
            idom: parent, // provisional
            sdom: NODEINFO_UNINIT,
            ancestor: NODEINFO_UNINIT,
        };
        self.dfs.push(cur_node);
        // NB assumes that CFG is fully connected.
        for n in self.cfg.neighbors_directed(cur_node, Direction::Outgoing) {
            if self.seen() == self.num_nodes() {
                break;
            }
            if self.at(n).seen() {
                continue;
            }
            self.dfs(n, cur_node.index() as NumTy);
        }
    }

    // Compute semidominators.
    // Assumes self.dfs has been called.
    fn semis(&mut self) {
        // We need to borrow self.dfs, but also other parts of the struct, so we swap it out.
        // Note that this only works because we know that `eval` and `link` do not use the
        // `dfs` vector.
        let dfs = std::mem::replace(&mut self.dfs, Default::default());
        for n in dfs[1..].iter().rev().map(|x| *x) {
            let parent = self.at(n).parent;
            let mut semi = parent;
            for pred in self
                .cfg
                .neighbors_directed(NodeIx::new(n.ix()), Direction::Incoming)
            {
                let candidate = if self.at(pred).dfsnum <= self.at(n).dfsnum {
                    pred.num()
                } else {
                    let ancestor_with_lowest = self.eval(pred);
                    self.at(ancestor_with_lowest).sdom
                };
                if self.at(candidate).dfsnum < self.at(semi).dfsnum {
                    semi = candidate
                }
            }
            *(&mut self.at_mut(n).sdom) = semi;
            self.link(parent, n);
        }
        std::mem::replace(&mut self.dfs, dfs);
    }

    // Compute dominator tree.
    // Assumes self.semis has been called.
    fn idoms(&mut self) {
        let dfs = std::mem::replace(&mut self.dfs, Default::default());
        for n in dfs[1..].iter().map(|x| *x) {
            let (mut idom, semi_dfs) = {
                let entry = self.at(n);
                (entry.idom, self.at(entry.sdom).dfsnum)
            };
            while self.at(idom).dfsnum > semi_dfs {
                idom = self.at(idom).idom;
            }
            (&mut self.at_mut(n)).idom = idom;
        }
        std::mem::replace(&mut self.dfs, dfs);
    }

    /// Compute the dominance fromtier.
    pub fn dom_frontier(&self) -> Frontier {
        // TODO: Add custom hash set and hash maps for dense integer keys.
        let mut fronts = vec![HashSet::<NumTy>::default(); self.info.len()];
        for (b_ix, b) in self.info.iter().enumerate() {
            let b_ix = b_ix as NumTy;
            let neighs: SmallVec<[NodeIx; 4]> = self
                .cfg
                .neighbors_directed(NodeIx::new(b_ix.ix()), Direction::Incoming)
                .collect();
            let bdom = b.idom;
            if neighs.len() >= 2 {
                for p in neighs.into_iter() {
                    let mut runner = p.num();
                    while runner != bdom {
                        fronts[runner.ix()].insert(b_ix);
                        runner = self.at(runner).idom;
                    }
                }
            }
        }
        fronts
    }

    /// Compute the dominator tree.
    pub fn dom_tree(&self) -> Tree {
        // All the information for the tree that we need has already been computed in the
        // initialization phase by `self.idoms`, but the tree is inverted. This method copies the tree
        // out rightside up.
        let mut tree = vec![SmallVec::new(); self.info.len()];
        for (i, info) in self.info.iter().enumerate() {
            if info.idom != NODEINFO_UNINIT {
                tree[info.idom as usize].push(i as NumTy)
            }
        }
        tree
    }

    // Short helper methods

    fn num_nodes(&self) -> NumTy {
        debug_assert_eq!(self.cfg.node_count(), self.info.len());
        self.info.len() as NumTy
    }
    fn seen(&self) -> NumTy {
        self.dfs.len() as NumTy
    }
    // TODO: Explore performance impact of performing checked indexing here and elsewhere. Is it
    // worth using unsafe or building a safe index API?
    fn at(&self, ix: impl HasNum) -> &NodeInfo {
        &self.info[ix.ix()]
    }
    fn best_at(&self, ix: impl HasNum) -> NumTy {
        self.best[ix.ix()]
    }
    fn set_best(&mut self, n: impl HasNum, v: impl HasNum) {
        self.best[n.ix()] = v.num();
    }
    fn at_mut(&mut self, ix: impl HasNum) -> &mut NodeInfo {
        &mut self.info[ix.ix()]
    }
    fn link(&mut self, parent: impl HasNum, node: impl HasNum) {
        *(&mut self.at_mut(node).ancestor) = parent.num();
        self.set_best(node, node);
    }
}

/// A utility trait making it easier to write functions that are polymorphic on index type.
trait HasNum: Copy {
    #[inline]
    fn ix(self) -> usize {
        self.num() as usize
    }

    #[inline]
    fn num(self) -> NumTy {
        const _MAX_NUMTY: NumTy = !0;
        debug_assert!((_MAX_NUMTY as usize) >= self.ix());
        self.ix() as NumTy
    }
}

impl HasNum for NumTy {
    #[inline]
    fn num(self) -> NumTy {
        self
    }
}

impl HasNum for NodeIx {
    #[inline]
    fn ix(self) -> usize {
        self.index()
    }
}

#[derive(Clone, Debug)]
struct NodeInfo {
    // order reached in DFS.
    dfsnum: NumTy,
    // immediate dominator
    idom: NumTy,
    // semidominator,
    sdom: NumTy,
    // parent in spanning tree
    parent: NumTy,
    // ancestor in spanning forest during semidominator calculation.
    // TODO: The LLVM implementation combines parents and ancestors, and instead passes in a
    // "Last Visited" dfsnum into `eval` in order to skip nodes that haven't been linked into
    // the tree. Make that change once confident in the initial one.
    ancestor: NumTy,
}

impl Default for NodeInfo {
    fn default() -> NodeInfo {
        NodeInfo {
            dfsnum: NODEINFO_UNINIT,
            idom: NODEINFO_UNINIT,
            sdom: NODEINFO_UNINIT,
            parent: NODEINFO_UNINIT,
            ancestor: NODEINFO_UNINIT,
        }
    }
}

impl NodeInfo {
    fn seen(&self) -> bool {
        self.dfsnum != NODEINFO_UNINIT
    }
}

const NODEINFO_UNINIT: NumTy = !0;

#[cfg(test)]
mod test {
    use super::*;

    // The dominator logic is written against a full Context; this helper function creates a dummy
    // context around a vector of edges specifying a graph. The edges must create a connected
    // graph.

    fn make_cfg_impl(edges: Vec<(NumTy, NumTy)>) -> (Graph<(), ()>, NodeIx) {
        // we assume that node 0 is going to be the entry node.
        let n_nodes = 1 + edges
            .iter()
            .flat_map(|(i, j)| vec![*i, *j].into_iter())
            .max()
            .unwrap();
        let mut cfg = Graph::<(), ()>::new();
        let ixes: Vec<_> = (0..n_nodes)
            .map(|_| cfg.add_node(Default::default()))
            .collect();
        cfg.extend_with_edges(edges.into_iter().map(|(i, j)| (ixes[i.ix()], ixes[j.ix()])));
        (cfg, NodeIx::new(0))
    }

    // In an effort to port over some test cases from the tiger book verbatim, here are some short
    // macros that let us use capital letters to specify graphs.

    macro_rules! table {
        ($x:tt) => {{
            let s = stringify!($x);
            let fst = s.as_bytes()[0] as usize;
            if fst > 90 || fst < 65 {
                assert!(
                    false,
                    "invalid identifier {:?}, need single character between 'A' and 'Z'",
                    s
                )
            }
            ((s.as_bytes()[0] as usize) - 65) as NumTy
        }};
    }

    fn table_inv(n: NumTy) -> String {
        (65..91)
            .nth(n as usize)
            .and_then(std::char::from_u32)
            .unwrap()
            .to_string()
    }

    macro_rules! make_cfg {
        ( $( $i:tt => $j:tt, )* ) => {
            make_cfg_impl(vec![ $( (table!($i), table!($j)) ),* ])
        }
    }

    macro_rules! check_tree {
        ($bld:expr, $fld:tt, $( $i:tt => $j:tt, )*) => {
            let b = $bld;
            $(
                let i = table!($i);
                let jdom = b.info[table!($j) as usize].$fld;
                assert_eq!(
                    jdom,
                    i,
                    "Expected {} of {:?}({}) to be {:?}({}), instead it was {:?}({})",
                    stringify!($fld),
                    stringify!($j),
                    table!($j),
                    stringify!($i),
                    i,
                    table_inv(jdom),
                    jdom
                );
            )*
        }
    }
    #[test]
    fn dom_frontier_calc() {
        let (cfg, entry) = make_cfg_impl(vec![
            (0, 1),
            (0, 4),
            (0, 8),
            (1, 2),
            (2, 2),
            (2, 3),
            (3, 12),
            (4, 5),
            (4, 6),
            (5, 7),
            (5, 3),
            (6, 7),
            (6, 11),
            (7, 4),
            (7, 12),
            (8, 9),
            (8, 10),
            (9, 11),
            (10, 11),
            (11, 12),
        ]);
        let fronts = DomInfo::new(&cfg, entry).dom_frontier();
        assert_eq!(fronts[0], Default::default());
        assert_eq!(fronts[1], vec![3].into_iter().collect());
        assert_eq!(fronts[2], vec![2, 3].into_iter().collect());
        assert_eq!(fronts[3], vec![12].into_iter().collect());
        assert_eq!(fronts[4], vec![3, 4, 11, 12].into_iter().collect());
        assert_eq!(fronts[5], vec![3, 7].into_iter().collect());
    }

    #[test]
    fn dom_tree() {
        // Tiger Book, Figure 19.8
        // Note that the semidominators will not necessarily be the same as they vary based on the
        // order in which successor nodes are visited.
        let (cfg, entry) = make_cfg! {
            A => B, A => C,
            B => D, B => G,
            C => E, C => H,
            D => G, D => F,
            E => C, E => H,
            F => I, F => K,
            G => J,
            H => M,
            I => L,
            J => I,
            K => L,
            L => M, L => B,
        };
        let builder = DomInfo::new(&cfg, entry);

        // In this case, semidominators and immediate dominators are the same.
        check_tree! { &builder, sdom,
            A => B, A => M, A => C,
            B => D, B => G, B => I, B => L,
            C => E, C => H,
            D => F,
            F => K,
            G => J,
        };
        check_tree! { &builder, idom,
            A => B, A => M, A => C,
            B => D, B => G, B => I, B => L,
            C => E, C => H,
            D => F,
            F => K,
            G => J,
        };
    }
}
