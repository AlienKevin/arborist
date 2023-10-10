use crate::action::ActionTrace;
use crate::io_pair::IOPair;
use crate::selector::Selector;
use crate::state::NIL_STATE_ID;
use crate::state_factory::{
    self, copy_state, get_search_tree_transition, get_state_by_id, increment_annotation,
};
use crate::synthesis::MAX_LOOP_BODY_DEPTH;
use crate::value_path::ValuePath;

use super::transition::{Transition, Transitions};
use super::{
    dsl,
    grammar_symbol::*,
    state::{dummy_state_id, StateId},
    state_factory::create_new_state,
};
use im_rc::{HashSet, Vector};
use itertools::Itertools;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

/// An FTA with states and transitions
#[derive(Clone)]
pub struct Fta {
    pub root: StateId,
    pub transitions: Transitions,
}

impl std::fmt::Display for Fta {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f, "digraph Fta {{")?;
        writeln!(f, "  compound=true;")?;
        writeln!(f, "  rankdir=BT;")?;
        writeln!(f, "  ranksep=\"1 equally\";")?;
        writeln!(f, "  node [ordering=\"out\"]")?;
        writeln!(f, "  edge [ordering=\"out\"]")?;
        // Show all the states as node clusters
        for state in self.bfs() {
            let transitions = self.transitions.get_by_id(state).unwrap();
            let num_transitions = transitions.len();
            if num_transitions > 1 && matches!(transitions.iter().next(), Some(Transition::Seq(..)))
            {
                // compound state
                writeln!(f, "subgraph cluster_{state} {{")?;
                writeln!(f, "  style=filled;")?;
                writeln!(f, "  color=lightgray;")?;
                writeln!(
                    f,
                    "  node [style=filled, fontcolor=white, color={color}, ordering=\"out\"];",
                    color = if self.root == state {
                        "blue"
                    } else {
                        "darkgray"
                    }
                )?;
                for i in 0..num_transitions {
                    writeln!(f, "  q{state}_{i} [label={state}];")?;
                }
                writeln!(f, "}}")?;
            } else if !matches!(transitions.iter().next(), Some(Transition::Nil)) {
                // simple state
                writeln!(f, "subgraph cluster_{state} {{")?;
                writeln!(f, "  style=invis;")?;
                match transitions.iter().next() {
                    Some(Transition::Seq(..) | Transition::Nil) => {
                        writeln!(
                            f,
                            "  q{state}_0 [label={state}, style=filled, fontcolor=white, color={color}];",
                            color = if self.root == state {
                                "blue"
                            } else {
                                "darkgray"
                            }
                        )?;
                    }
                    _ => {
                        writeln!(f, "  q{state}_0 [label={state}];")?;
                    }
                }
                writeln!(f, "}}")?;
            }
        }
        // Show all the transitions as edges
        for state in self.bfs() {
            for (i, transition) in self
                .transitions
                .get_by_id(state)
                .unwrap()
                .iter()
                .enumerate()
            {
                match transition {
                    Transition::Seq(q0, q1) => {
                        writeln!(
                            f,
                            "q{q0}_0 -> q{state}_{i} [label=\"L\", penwidth=2, style=dashed{ltail}];",
                            ltail = if self.is_simple_state(*q0) {
                                String::new()
                            } else {
                                format!(", ltail=cluster_{}", q0)
                            }
                        )?;
                        let is_right_branch_nil = *q1 == NIL_STATE_ID;
                        if is_right_branch_nil {
                            writeln!(f, "q{q0}_{i}_nil [style=invis];")?;
                        }
                        writeln!(
                            f,
                            "{source} -> q{state}_{i} [label=\"{label}\", penwidth=2{ltail}];",
                            source = if is_right_branch_nil {
                                format!("q{}_{}_nil", q0, i)
                            } else {
                                format!("q{}_0", q1)
                            },
                            label = if is_right_branch_nil { "Nil" } else { "R" },
                            ltail = if self.is_simple_state(*q1) {
                                String::new()
                            } else {
                                format!(", ltail=cluster_{}", q1)
                            }
                        )?;
                    }
                    Transition::Nil => {
                        // // invisible start node
                        // writeln!(f, "q{}_{}_ [style=invis];", state, i)?;
                        // writeln!(
                        //     f,
                        //     "q{}_{}_ -> q{}_{} [label=\" {}\", penwidth={}];",
                        //     state, i, state, i, "Nil", "2"
                        // )?;

                        // Skip Nil to avoid overwhelming the GraphViz visualization
                    }
                    Transition::Nullary(op) => {
                        // invisible start node
                        writeln!(f, "q{state}_{i}_ [style=invis];")?;
                        writeln!(
                            f,
                            "q{state}_{i}_ -> q{state}_{i} [label=\"{op}\", penwidth=\"1\"];",
                        )?;
                    }
                    Transition::Unary(op, ns) => {
                        // invisible start node
                        writeln!(f, "q{state}_{i}_ [style=invis];")?;
                        writeln!(
                            f,
                            "q{state}_{i}_ -> q{state}_{i} [label=\"{op} ({num_ns})\", penwidth=\"1\", labeltooltip=\"{tooltip}\"];",
                            num_ns = ns.len(),
                            tooltip = ns.iter().map(|n| format!("{}", n)).join("\n"),
                        )?;
                    }
                    Transition::For(q_body) => {
                        // Hide the rest of the For transitions
                        // if i > 0 {
                        //     continue;
                        // }
                        let tooltip = get_state_by_id(state)
                            .annotations
                            .last()
                            .map(|annotation| annotation.to_string())
                            .unwrap_or_else(|| "No annotations".to_string());
                        writeln!(
                            f,
                            "q{q_body}_0 -> q{state}_{i} [label=<<table border=\"0\" cellborder=\"0\">\
                                <tr><td bgcolor=\"orange\"><B>For</B></td></tr>\
                            </table>>, penwidth=\"1\", labeltooltip=\"{tooltip}\"{ltail}];",
                            tooltip = tooltip,
                            ltail = if self.is_simple_state(*q_body) {
                                String::new()
                            } else {
                                format!(", ltail=cluster_{}", q_body)
                            },
                        )?;
                    }
                    // TODO: Refactor duplicated While and For display logic
                    Transition::While(q_body, ns) => {
                        let tooltip = get_state_by_id(state)
                            .annotations
                            .last()
                            .map(|annotation| annotation.to_string())
                            .unwrap_or_else(|| "No annotations".to_string());
                        writeln!(
                            f,
                            "q{q_body}_0 -> q{state}_{i} [label=<<table border=\"0\" cellborder=\"0\">\
                                <tr><td bgcolor=\"orange\"><B>While({num_ns})</B></td></tr>\
                            </table>>, penwidth=\"1\", labeltooltip=\"{tooltip}\"{ltail}];",
                            num_ns = ns.len(),
                            tooltip = tooltip,
                            ltail = if self.is_simple_state(*q_body) {
                                String::new()
                            } else {
                                format!(", ltail=cluster_{}", q_body)
                            },
                        )?;
                    }
                    Transition::ForData(vp, q_body) => {
                        let tooltip = get_state_by_id(state)
                            .annotations
                            .last()
                            .map(|annotation| annotation.to_string())
                            .unwrap_or_else(|| "No annotations".to_string());
                        writeln!(
                            f,
                            "q{q_body}_0 -> q{state}_{i} [label=<<table border=\"0\" cellborder=\"0\">\
                                <tr><td bgcolor=\"orange\"><B>ForData({vp})</B></td></tr>\
                            </table>>, penwidth=\"1\", labeltooltip=\"{tooltip}\"{ltail}];",
                            vp = vp,
                            tooltip = tooltip,
                            ltail = if self.is_simple_state(*q_body) {
                                String::new()
                            } else {
                                format!(", ltail=cluster_{}", q_body)
                            },
                        )?;
                    }
                    Transition::SendKeys(ns, _s) => {
                        // invisible start node
                        writeln!(f, "q{state}_{i}_ [style=invis];")?;
                        writeln!(
                            f,
                            "q{state}_{i}_ -> q{state}_{i} [label=\"SendKeys ({num_ns})\", penwidth=\"1\", labeltooltip=\"{tooltip}\"];",
                            num_ns = ns.len(),
                            tooltip = ns.iter().map(|n| format!("{}", n)).join("\n"),
                        )?;
                    }
                    Transition::SendData(ns, _v) => {
                        // invisible start node
                        writeln!(f, "q{state}_{i}_ [style=invis];")?;
                        writeln!(
                            f,
                            "q{state}_{i}_ -> q{state}_{i} [label=\"SendData ({num_ns})\", penwidth=\"1\", labeltooltip=\"{tooltip}\"];",
                            num_ns = ns.len(),
                            tooltip = ns.iter().map(|n| format!("{}", n)).join("\n"),
                        )?;
                    }
                }
            }
        }
        writeln!(f, "}}")
    }
}

impl From<dsl::Program> for Fta {
    fn from(program: dsl::Program) -> Self {
        let (mut fta, root_seq_state) = Fta::init_root_seq();
        fta.from_program(&program, root_seq_state);
        fta
    }
}

impl From<dsl::GroupedProgram> for Fta {
    fn from(program: dsl::GroupedProgram) -> Self {
        let (mut fta, root_seq_state) = Fta::init_root_seq();
        fta.from_grouped_program(&program, root_seq_state);
        fta
    }
}

impl From<dsl::GroupedPrograms> for Fta {
    fn from(programs: dsl::GroupedPrograms) -> Self {
        let (mut fta, mut root_state) = Fta::init_root_seq();
        for (i, program) in programs.iter().enumerate() {
            fta.from_grouped_program(program, root_state);
            if i < programs.len() - 1 {
                root_state = fta.create_final_state();
            } else {
                break;
            }
        }
        fta
    }
}

#[derive(PartialEq, Eq, Clone, Copy)]
pub enum IgnoreLoop {
    Ignore,
    Allow,
}

#[derive(Clone)]
pub struct FtaBfsIterator<'a> {
    fta: &'a Fta,
    visited: FxHashSet<StateId>,
    queue: VecDeque<StateId>,
}

impl<'a> FtaBfsIterator<'a> {
    fn visit(&mut self, state: StateId) -> bool {
        if self.visited.contains(&state) {
            true
        } else {
            self.visited.insert(state);
            false
        }
    }
}

impl<'a> Iterator for FtaBfsIterator<'a> {
    type Item = StateId;

    fn next(&mut self) -> Option<Self::Item> {
        let state = self.queue.pop_front()?;
        let transitions = match self.fta.transitions.get_by_id(state) {
            Some(transitions) => transitions,
            None => panic!("State with id {} has no input transitions.", state),
        };
        for (_i, transition) in transitions.iter().enumerate() {
            use Transition::*;
            let children_states = match &transition {
                &Seq(q0, q1) => {
                    vec![*q0, *q1]
                }
                &For(q2) => {
                    // Hide rest of the For transitions
                    // if i > 0 {
                    //     continue;
                    // }
                    vec![*q2]
                }
                &While(q0, _) => {
                    vec![*q0]
                }
                &ForData(_, q2) => {
                    vec![*q2]
                }
                &Nil | &Nullary(_) | &Unary(..) | &SendKeys(..) | &SendData(..) => continue,
            };
            for q in children_states {
                if self.visit(q) {
                    continue;
                } else {
                    self.queue.push_back(q);
                }
            }
        }
        Some(state)
    }
}

pub enum CopyDepth {
    DeepCopy,
    ShallowCopy,
}

impl Default for Fta {
    fn default() -> Self {
        crate::state_factory::clear_state_factory();

        let mut fta = Self {
            root: 0,
            transitions: Transitions::empty(),
        };
        fta.create_nil_state();

        fta
    }
}

impl Fta {
    fn is_simple_state(&self, state_id: StateId) -> bool {
        let transitions = self.transitions.get_by_id(state_id).unwrap();
        transitions.len() == 1 || matches!(transitions.iter().next(), Some(Transition::For(..)))
    }

    pub fn new(root: StateId, transitions: Transitions) -> Self {
        Self { root, transitions }
    }

    fn init_root_seq() -> (Self, StateId) {
        crate::state_factory::clear_state_factory();

        let mut fta = Self {
            root: 0,
            transitions: Transitions::empty(),
        };
        fta.create_nil_state();
        let root_seq_state = fta.create_final_state();
        (fta, root_seq_state)
    }

    // Assumes that the FTA is initialized properly
    fn from_program(&mut self, program: &dsl::Program, root_seq_state: StateId) {
        let mut seq_state = root_seq_state;
        // Initialize seq_right_state with a dummy State
        // that is going to be overwritten in for_each
        let mut seq_right_state = dummy_state_id();
        program.iter().enumerate().for_each(|(i, expr)| {
            let seq_left_state = create_new_state(GrammarSymbol::Expr);
            seq_right_state = if i < program.len() - 1 {
                create_new_state(GrammarSymbol::Program)
            } else {
                NIL_STATE_ID
            };
            self.add_transition(Transition::Seq(seq_left_state, seq_right_state), seq_state);
            // set the next seq_state for the right branch of the current seq
            seq_state = seq_right_state;
            match expr {
                dsl::Expr::Nullary(op) => {
                    self.add_transition(Transition::Nullary(*op), seq_left_state);
                }
                dsl::Expr::Unary(ns, op) => {
                    self.add_transition(Transition::Unary(*op, ns.clone()), seq_left_state);
                }
                dsl::Expr::While(p, ns) => {
                    let while_p_state = create_new_state(GrammarSymbol::Program);
                    self.add_transition(
                        Transition::While(while_p_state, ns.clone()),
                        seq_left_state,
                    );
                    self.from_program(p, while_p_state);
                }
                dsl::Expr::For(p) => {
                    let for_p_state = create_new_state(GrammarSymbol::Program);
                    self.add_transition(Transition::For(for_p_state), seq_left_state);
                    self.from_program(p, for_p_state);
                }
                dsl::Expr::ForData(vp, p) => {
                    let for_p_state = create_new_state(GrammarSymbol::Program);
                    self.add_transition(
                        Transition::ForData(vp.clone(), for_p_state),
                        seq_left_state,
                    );
                    self.from_program(p, for_p_state);
                }
                dsl::Expr::SendKeys(ns, s) => {
                    self.add_transition(
                        Transition::SendKeys(ns.clone(), s.clone()),
                        seq_left_state,
                    );
                }
                dsl::Expr::SendData(ns, v) => {
                    self.add_transition(
                        Transition::SendData(ns.clone(), v.clone()),
                        seq_left_state,
                    );
                }
            }
        });
    }

    fn from_grouped_expr(&mut self, expr: &dsl::GroupedExpr, root_state: StateId) {
        use dsl::GroupedExpr::*;
        match expr {
            Nullary(op) => {
                self.add_transition(Transition::Nullary(*op), root_state);
            }
            Unary(ns, op) => {
                self.add_transition(Transition::Unary(*op, ns.clone()), root_state);
            }
            While(ps, ns) => {
                let while_p_state = create_new_state(GrammarSymbol::Program);
                self.add_transition(Transition::While(while_p_state, ns.clone()), root_state);
                for p in ps {
                    self.from_grouped_program(p, while_p_state);
                }
            }
            For(ps) => {
                let for_p_state = create_new_state(GrammarSymbol::Program);
                self.add_transition(Transition::For(for_p_state), root_state);
                for p in ps {
                    self.from_grouped_program(p, for_p_state);
                }
            }
            ForData(vp, ps) => {
                let for_p_state = create_new_state(GrammarSymbol::Program);
                self.add_transition(Transition::ForData(vp.clone(), for_p_state), root_state);
                for p in ps {
                    self.from_grouped_program(p, for_p_state);
                }
            }
            SendKeys(ns, str) => {
                self.add_transition(Transition::SendKeys(ns.clone(), str.clone()), root_state);
            }
            SendData(ns, v) => {
                self.add_transition(Transition::SendData(ns.clone(), v.clone()), root_state);
            }
        }
    }

    // Assumes that the FTA is initialized properly
    fn from_grouped_program(&mut self, program: &dsl::GroupedProgram, root_seq_state: StateId) {
        let seq_left_state = create_new_state(GrammarSymbol::Expr);
        let seq_right_state = if let dsl::GroupedProgram::GroupedSeq(..) = program {
            create_new_state(GrammarSymbol::Program)
        } else {
            NIL_STATE_ID
        };
        self.add_transition(
            Transition::Seq(seq_left_state, seq_right_state),
            root_seq_state,
        );
        match program {
            dsl::GroupedProgram::GroupedSeq(es, ps) => {
                for e in es {
                    self.from_grouped_expr(e, seq_left_state);
                }
                for p in ps {
                    self.from_grouped_program(p, seq_right_state);
                }
            }
            dsl::GroupedProgram::GroupedSingle(es) => {
                for e in es {
                    self.from_grouped_expr(e, seq_left_state);
                }
            }
        }
    }

    pub fn bfs(&self) -> FtaBfsIterator {
        FtaBfsIterator {
            fta: self,
            visited: FxHashSet::default(),
            queue: VecDeque::from([self.root]),
        }
    }

    pub fn is_over_max_body_depth(&self, q: StateId) -> bool {
        let mut depth = 0;
        use crate::fta::Transition::*;
        for transition in self.transitions.get_by_id(q).unwrap() {
            match transition {
                Nullary(_) | Nil | Unary(..) | SendKeys(..) | SendData(..) => {
                    // do nothing
                }
                While(q_body, _) | For(q_body) | ForData(_, q_body) => {
                    let body_depth = self.get_depth(*q_body);
                    depth = std::cmp::max(body_depth + 1, depth);
                    if depth > MAX_LOOP_BODY_DEPTH {
                        return true;
                    }
                }
                &Seq(q1, q2) => {
                    let q1_depth = self.get_depth(q1);
                    if q1_depth > MAX_LOOP_BODY_DEPTH {
                        return true;
                    }
                    let q2_depth = self.get_depth(q2);
                    if q1_depth > MAX_LOOP_BODY_DEPTH {
                        return true;
                    }
                    depth = std::cmp::max(q1_depth, q2_depth);
                }
            }
        }

        depth > MAX_LOOP_BODY_DEPTH
    }

    pub fn get_depth(&self, q: StateId) -> usize {
        let mut depth = 0;
        use crate::fta::Transition::*;
        for transition in self.transitions.get_by_id(q).unwrap() {
            match transition {
                Nullary(_) | Nil | Unary(..) | SendKeys(..) | SendData(..) => {
                    // do nothing
                }
                While(q_body, _) | For(q_body) | ForData(_, q_body) => {
                    let body_depth = self.get_depth(*q_body);
                    depth = std::cmp::max(body_depth + 1, depth);
                }
                &Seq(q1, q2) => {
                    let q1_depth = self.get_depth(q1);
                    let q2_depth = self.get_depth(q2);
                    depth = std::cmp::max(q1_depth, q2_depth);
                }
            }
        }
        depth
    }

    pub fn deep_copy(&self, q: StateId, is_state_copied: bool) -> (StateId, Transitions) {
        // Do not copy shared Nil state
        let q_ = if q == NIL_STATE_ID || !is_state_copied {
            q
        } else {
            copy_state(q)
        };
        let mut transitions_ = Transitions::empty();
        use crate::fta::Transition::*;
        for transition in self.transitions.get_by_id(q).unwrap() {
            match transition {
                Nullary(_) | Nil => {
                    transitions_ = transitions_.insert(transition.clone(), q_);
                }
                Unary(op, ns) => {
                    transitions_ = transitions_.insert(Unary(*op, ns.clone()), q_);
                }
                While(q1, ns) => {
                    let (q1_, transitions1_) = self.deep_copy(*q1, is_state_copied);
                    transitions_ = transitions_
                        .union(transitions1_)
                        .insert(While(q1_, ns.clone()), q_);
                }
                &Seq(q1, q2) => {
                    let (q1_, transitions1_) = self.deep_copy(q1, is_state_copied);
                    let (q2_, transitions2_) = self.deep_copy(q2, is_state_copied);
                    transitions_ = transitions_
                        .union(transitions1_)
                        .union(transitions2_)
                        .insert(Seq(q1_, q2_), q_);
                }
                For(q3) => {
                    let (q3_, transitions3_) = self.deep_copy(*q3, is_state_copied);
                    transitions_ = transitions_.union(transitions3_).insert(For(q3_), q_);
                }
                SendKeys(ns, s) => {
                    transitions_ = transitions_.insert(SendKeys(ns.clone(), s.clone()), q_);
                }
                SendData(ns, v) => {
                    transitions_ = transitions_.insert(SendData(ns.clone(), v.clone()), q_);
                }
                ForData(vp, q3) => {
                    let (q3_, transitions3_) = self.deep_copy(*q3, is_state_copied);
                    transitions_ = transitions_
                        .union(transitions3_)
                        .insert(ForData(vp.clone(), q3_), q_);
                }
            }
        }
        (q_, transitions_)
    }

    pub fn copy_body(
        &self,
        q0: StateId,
        pqs: &[(StateId, StateId)],
        copy_depth: CopyDepth,
    ) -> Transitions {
        let (transitions_, _) = pqs.iter().enumerate().fold(
            (Transitions::empty(), q0),
            |(mut transitions, q0), (i, (p1, q1))| {
                let q0_ = state_factory::copy_state(q0);
                let p1_ = match copy_depth {
                    CopyDepth::DeepCopy => {
                        let (p1_, p1_transitions) = self.deep_copy(*p1, true);
                        transitions.union_inplace(p1_transitions);
                        p1_
                    }
                    CopyDepth::ShallowCopy => state_factory::copy_state(*p1),
                };
                let q1_ = if i == pqs.len() - 1 {
                    transitions.insert_inplace(Transition::Nil, NIL_STATE_ID);
                    NIL_STATE_ID
                } else {
                    state_factory::copy_state(*q1)
                };
                transitions.insert_inplace(Transition::Seq(p1_, q1_), q0_);
                (transitions, *q1)
            },
        );
        transitions_
    }

    pub fn copy_fordata_body(
        &self,
        q0: StateId,
        pqs: &[(StateId, StateId)],
        ns_map: FxHashMap<StateId, (Vector<Selector>, ValuePath)>,
    ) -> Transitions {
        let (transitions_, _) = pqs.iter().enumerate().fold(
            (Transitions::empty(), q0),
            |(mut transitions, q0), (i, (p1, q1))| {
                let q0_ = state_factory::copy_state(q0);
                let p1_ = if let Some((ns, vp)) = ns_map.get(p1) {
                    let p1_ = state_factory::copy_state(*p1);
                    transitions.insert_inplace(Transition::SendData(ns.clone(), vp.clone()), p1_);
                    p1_
                } else {
                    let (p1_, p1_transitions) = self.deep_copy(*p1, true);
                    transitions.union_inplace(p1_transitions);
                    p1_
                };
                let q1_ = if i == pqs.len() - 1 {
                    transitions.insert_inplace(Transition::Nil, NIL_STATE_ID);
                    NIL_STATE_ID
                } else {
                    state_factory::copy_state(*q1)
                };
                transitions.insert_inplace(Transition::Seq(p1_, q1_), q0_);
                (transitions, *q1)
            },
        );
        transitions_
    }

    pub fn copy_while_body(
        &self,
        q0: StateId,
        pqs: &[(StateId, StateId)],
    ) -> (StateId, Transitions) {
        let ql = pqs.last().unwrap().1;
        let ql_annotations = get_state_by_id(ql).annotations;
        assert!(ql_annotations.len() == 1);
        let ql_action_index = ql_annotations[0].output.get_start();
        let mut copied_q0 = 0; // dummy placeholder value
        let (transitions_, _) = pqs.iter().enumerate().fold(
            (Transitions::empty(), q0),
            |(mut transitions, q0), (i, (p1, q1))| {
                let q0_annotations = get_state_by_id(q0).annotations;
                assert!(q0_annotations.len() == 1);
                let q0_start_action_index = q0_annotations[0].output.get_start();
                let q0_to_ql_action_trace = ActionTrace(q0_start_action_index, ql_action_index);
                let q0_ = state_factory::create_new_state_with_annotations(
                    GrammarSymbol::Program,
                    vec![IOPair {
                        input: None,
                        output: q0_to_ql_action_trace,
                    }],
                );
                if i == 0 {
                    copied_q0 = q0_;
                }
                let p1_ = {
                    let (p1_, p1_transitions) = self.deep_copy(*p1, false);
                    transitions.union_inplace(p1_transitions);
                    p1_
                };
                let q1_ = if i == pqs.len() - 1 {
                    transitions.insert_inplace(Transition::Nil, NIL_STATE_ID);
                    NIL_STATE_ID
                } else {
                    let q1_annotations = get_state_by_id(*q1).annotations;
                    assert!(q1_annotations.len() == 1);
                    let q1_start_action_index = q1_annotations[0].output.get_start();
                    let q1_to_ql_action_trace = ActionTrace(q1_start_action_index, ql_action_index);

                    state_factory::create_new_state_with_annotations(
                        GrammarSymbol::Program,
                        vec![IOPair {
                            input: None,
                            output: q1_to_ql_action_trace,
                        }],
                    )
                };
                transitions.insert_inplace(Transition::Seq(p1_, q1_), q0_);
                (transitions, *q1)
            },
        );
        (copied_q0, transitions_)
    }

    pub fn clean_up(&mut self, root: StateId) {
        let result_transitions = self.clean_up_return_transitions(root);
        self.transitions = result_transitions;
    }

    fn clean_up_return_transitions(&mut self, root: StateId) -> Transitions {
        let mut result_transitions = Transitions::empty();
        let mut states = VecDeque::from([root]);
        while !states.is_empty() {
            let state = states.pop_front().unwrap();
            let transitions = match self.transitions.0.remove(&state) {
                Some(transitions) => transitions,
                None => continue,
            };

            for transition in transitions {
                use Transition::*;
                match &transition {
                    &Seq(q0, q1) => {
                        states.extend([q0, q1]);
                    }
                    For(q2) => {
                        states.push_back(*q2);
                    }
                    ForData(_vp, q2) => {
                        states.push_back(*q2);
                    }
                    &While(q0, _) => {
                        states.push_back(q0);
                    }
                    &Nil | &Nullary(_) | &Unary(..) | SendKeys(..) | SendData(..) => {}
                }
                result_transitions = result_transitions.insert(transition, state);
            }
        }
        result_transitions
    }

    pub fn clean_up_insert_missing(&mut self, mut other: Fta) {
        let mut result_transitions = Transitions::empty();
        let mut states = VecDeque::from([self.root]);
        while !states.is_empty() {
            let state = states.pop_front().unwrap();
            let transitions = match self.transitions.0.remove(&state) {
                Some(transitions) => transitions,
                None => HashSet::default(),
            };

            for transition in transitions {
                use Transition::*;
                match &transition {
                    &Seq(q0, q1) => {
                        states.extend([q0, q1]);
                    }
                    For(q2) => {
                        states.push_back(*q2);
                    }
                    ForData(_vp, q2) => {
                        states.push_back(*q2);
                    }
                    &While(q0, _) => {
                        states.push_back(q0);
                    }
                    &Nil | &Nullary(_) | &Unary(..) | SendKeys(..) | SendData(..) => {}
                }
                result_transitions = result_transitions.insert(transition, state);
            }

            let other_transitions = other.clean_up_return_transitions(state);
            if !other_transitions.0.is_empty() {
                result_transitions.union_inplace(other_transitions);
            }
        }
        self.transitions = result_transitions;
    }

    pub fn search_top_level_seq(
        target: StateId,
        right: StateId,
    ) -> Vec<(Vec<(StateId, StateId)>, StateId)> {
        if let Some((top_level, _)) = get_search_tree_transition(right) {
            if top_level.target == target {
                vec![(vec![(top_level.left, right)], target)]
            } else {
                Self::search_top_level_seq(target, top_level.target)
                    .into_iter()
                    .map(|(mut result_pqs, result_right)| {
                        result_pqs.push((top_level.left, right));
                        (result_pqs, result_right)
                    })
                    .collect_vec()
            }
        } else {
            vec![]
        }
    }

    pub fn search_seq_of_len_top_down(
        &self,
        l: usize,
        start_id: StateId,
        ignore_loop: IgnoreLoop,
    ) -> Vec<(Vec<(StateId, StateId)>, StateId)> {
        self.transitions
            .get_by_id(start_id)
            .unwrap()
            .iter()
            .flat_map(|seq| match seq {
                Transition::Nil => vec![],
                &Transition::Seq(left, right) => {
                    // TODO: Handle ignore_loop for other kinds of loops like while and for value loops
                    let terminate = ignore_loop == IgnoreLoop::Ignore
                        && self
                            .transitions
                            .get_by_id(left)
                            .unwrap()
                            .iter()
                            .any(|transition| matches!(transition, Transition::For(..)));
                    if l == 0 || terminate {
                        vec![]
                    } else if l == 1 {
                        vec![(vec![(left, right)], right)]
                    } else {
                        let rests = self.search_seq_of_len_top_down(l - 1, right, ignore_loop);
                        if rests.is_empty() {
                            vec![(vec![(left, right)], right)]
                        } else {
                            rests
                                .into_iter()
                                .map(|(mut body_states, next_iter_start)| {
                                    body_states.insert(0, (left, right));
                                    (body_states, next_iter_start)
                                })
                                .collect()
                        }
                    }
                }
                _ => panic!("Expecting top-level Seqs."),
            })
            .collect()
    }

    pub fn increment_annotations(&self) {
        let mut visited_states = FxHashSet::default();
        self.increment_annotations_helper(self.root, &mut visited_states);
        visited_states.insert(self.root);
    }

    fn increment_annotations_helper(&self, q: StateId, visited_states: &mut FxHashSet<StateId>) {
        if visited_states.contains(&q) || q == NIL_STATE_ID {
            return;
        }
        visited_states.insert(q);
        increment_annotation(q);
        // println!("incred annotation of state {:?} to {:?}", q, get_state_by_id(q).annotations);
        for transition in self.transitions.get_by_id(q).unwrap().iter() {
            if let Transition::Seq(_left, right) = transition {
                self.increment_annotations_helper(*right, visited_states);
            }
        }
    }

    /// Create a new final state
    ///
    /// Used in testing
    fn create_final_state(&mut self) -> StateId {
        let final_state = create_new_state(GrammarSymbol::Program);
        self.root = final_state;
        final_state
    }

    fn create_nil_state(&mut self) {
        let nil_state_id = create_new_state(GrammarSymbol::Program);
        assert!(nil_state_id == NIL_STATE_ID);
        self.add_transition(Transition::Nil, nil_state_id);
    }

    /// Add a new transition to our FTA
    pub fn add_transition(&mut self, input_alphabet: Transition, output_state: StateId) {
        let transitions = std::mem::take(&mut self.transitions);
        self.transitions = transitions.insert(input_alphabet, output_state);
    }
}
