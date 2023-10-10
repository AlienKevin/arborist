use crate::{
    action::{Action, ActionTrace, Actions},
    dsl::{program_to_string, NullaryOp, UnaryOp},
    fta::{CopyDepth, Fta},
    grammar_symbol::GrammarSymbol,
    io_pair::IOPair,
    selector::{find_intersection_selectors, Index, Selector},
    selector_matrix::Matrices,
    state::{StateId, NIL_STATE_ID},
    state_factory::{
        self, add_search_tree_loop_transition, copy_state, get_first_while_pattern,
        get_is_same_action_type, get_last_output, get_search_tree_transition, get_state_by_id,
        set_first_while_pattern, set_is_same_action_type,
    },
    test_utils::{BenchmarkConfig, TimeStats},
    transition::{Transition, Transitions},
    value_path::{anti_unify_paths, ValuePath},
};

use im_rc::Vector;
use itertools::Itertools;
use rustc_hash::FxHashMap;
use serde_json::Value;
use std::collections::VecDeque;
use std::{
    sync::atomic::{AtomicUsize, Ordering},
    time::{Duration, Instant},
};

pub const MAX_LOOP_BODY_DEPTH: usize = 4;
const MAX_LOOP_BODY_LENGTH: usize = 10;

pub fn synthesis(
    fta: &mut Fta,
    actions: &Actions,
    matrices: &Matrices,
    data: &Option<Value>,
    config: &BenchmarkConfig,
    time_left: Duration,
    time_stats: &mut TimeStats,
) {
    for i in 0..1 {
        synthesis_once(
            fta, actions, matrices, data, i, config, time_left, time_stats,
        );
    }
}

struct ForCandidate {
    q0: StateId,
    l: usize,
    for_fta: Fta,
}

struct WhileCandidate {
    q0: StateId,
    pqs1: Vec<(StateId, StateId)>,
    ql: StateId,
    ns: Vector<Selector>,
}

struct FordataCandidate {
    q0: StateId,
    pqs1: Vec<(StateId, StateId)>,
    ns_map: FxHashMap<StateId, (Vector<Selector>, ValuePath)>,
    vp: ValuePath,
}

fn match_candidates(
    fta: &mut Fta,
    data: &Option<Value>,
    for_candidates: &mut Vec<ForCandidate>,
    while_candidates: &mut Vec<WhileCandidate>,
    fordata_candidates: &mut Vec<FordataCandidate>,
    pqs: &[(StateId, StateId)],
    q0: StateId,
    max_big_n_start_index: Index,
) {
    let l = pqs.len() / 2;
    let ql = pqs[l - 1].1;
    let pqs1 = &pqs[..l];
    let pqs2 = &pqs[l..];
    let mut has_senddata = false;
    for ((p1, _), (p2, _)) in pqs1.iter().zip(pqs2) {
        if !is_same_action_type(fta, *p1, *p2, &mut has_senddata) {
            return;
        }
    }

    if !has_senddata {
        if let Some(for_fta) = parametrize_for(fta, q0, pqs1, pqs2, max_big_n_start_index) {
            let candidate = ForCandidate { q0, l, for_fta };
            for_candidates.push(candidate);
        }
    }

    // only synthesize when there is data related work
    if data.is_some() {
        let mut shared_vp = None;
        let mut ns_map = FxHashMap::default();
        for (pq1, pq2) in pqs1.iter().zip(pqs2.iter()) {
            let &(p1, _) = pq1;
            let &(p2, _) = pq2;
            if let Transition::SendData(ns1, vp1) = fta
                .transitions
                .get_by_id(p1)
                .unwrap()
                .iter()
                .next()
                .unwrap()
            {
                if let Transition::SendData(ns2, vp2) = fta
                    .transitions
                    .get_by_id(p2)
                    .unwrap()
                    .iter()
                    .next()
                    .unwrap()
                {
                    if let Some(vp) = anti_unify_paths(vp1, vp2) {
                        if shared_vp.is_none() || &vp == shared_vp.as_ref().unwrap() {
                            shared_vp = Some(vp);
                            let ns = find_intersection_selectors(ns1.clone(), ns2);
                            if ns.is_empty() {
                                ns_map.clear();
                                break;
                            } else {
                                ns_map.insert(p1, (ns, vp1.clone()));
                            }
                        } else {
                            ns_map.clear();
                            break;
                        }
                    }
                }
            }
        }
        if !ns_map.is_empty() {
            if let Some(vp) = shared_vp {
                let candidate = FordataCandidate {
                    q0,
                    pqs1: pqs1.to_vec(),
                    ns_map,
                    vp,
                };
                fordata_candidates.push(candidate);
            }
        }
    }

    // Prevent synthesizing Whiles with empty bodies
    if l >= 2 && pqs2.len() == l {
        let p1 = pqs1.last().unwrap().0;
        let p2 = pqs2.last().unwrap().0;
        let p1_transitions = fta.transitions.get_by_id(p1).unwrap();
        if p1_transitions.len() == 1 {
            if let Transition::Unary(UnaryOp::Click, ns1) = p1_transitions.iter().next().unwrap() {
                let p2_transitions = fta.transitions.get_by_id(p2).unwrap();
                if p2_transitions.len() == 1 {
                    if let Transition::Unary(UnaryOp::Click, ns2) =
                        p2_transitions.iter().next().unwrap()
                    {
                        let ns: Vector<_> =
                            ns1.iter().filter(|n1| ns2.contains(n1)).cloned().collect();
                        if !ns.is_empty() {
                            let mut skip_matching_while = false;
                            if let Some(reference_pattern) = get_first_while_pattern(q0) {
                                if let Some(query_pattern) =
                                    get_while_body_pattern_from_pqs(fta, pqs1)
                                {
                                    println!("query while pattern: {:?}", query_pattern);
                                    println!("reference while pattern: {:?}", reference_pattern);
                                    if matches_while_body_pattern(
                                        &query_pattern,
                                        &reference_pattern,
                                    ) {
                                        println!("Matched while pattern");
                                        skip_matching_while = true;
                                    }
                                }
                            }
                            if !skip_matching_while {
                                while_candidates.push(WhileCandidate {
                                    q0,
                                    pqs1: pqs1.to_vec(),
                                    ql,
                                    ns,
                                });
                            }
                        }
                    }
                }
            }
        }
    }
}

pub fn search_seq_of_len_bottom_up(
    l: usize,
    right: StateId,
    buffer: &mut VecDeque<(StateId, StateId)>,
    func: &mut impl FnMut(&mut VecDeque<(StateId, StateId)>, StateId),
) {
    if let Some((top_level, entries)) = get_search_tree_transition(right) {
        if l == 1 {
            for entry in entries.iter().chain(std::iter::once(&top_level)) {
                buffer.push_front((entry.left, right));
                func(buffer, entry.target);
                buffer.pop_front();
            }
        } else {
            for entry in entries.iter().chain(std::iter::once(&top_level)) {
                buffer.push_front((entry.left, right));
                search_seq_of_len_bottom_up(l - 1, entry.target, buffer, func);
                buffer.pop_front();
            }
        }
    }
}

pub fn synthesis_once(
    fta: &mut Fta,
    actions: &Actions,
    matrices: &Matrices,
    data: &Option<Value>,
    _i: usize,
    config: &BenchmarkConfig,
    time_left: Duration,
    time_stats: &mut TimeStats,
) {
    let start_time = Instant::now();

    let num_for_candidates = AtomicUsize::new(0);
    let num_fors_linked_to_middle = AtomicUsize::new(0);
    let num_fors_linked_to_last = AtomicUsize::new(0);

    let num_while_candidates = AtomicUsize::new(0);
    let num_whiles_linked_to_middle = AtomicUsize::new(0);
    let num_whiles_linked_to_last = AtomicUsize::new(0);

    let num_fordata_candidates = AtomicUsize::new(0);
    let num_fordatas_linked_to_middle = AtomicUsize::new(0);
    let num_fordatas_linked_to_last = AtomicUsize::new(0);

    // println!("Search tree size: {:?}", get_search_tree_size());

    let mut for_candidates = vec![];
    let mut while_candidates = vec![];
    let mut fordata_candidates = vec![];

    let mut speculated_transitions = Transitions::empty();
    for l in 1..=MAX_LOOP_BODY_LENGTH {
        if start_time.elapsed() >= time_left {
            return;
        }
        search_seq_of_len_bottom_up(l * 2, NIL_STATE_ID, &mut VecDeque::new(), &mut |pqs, q0| {
            match_candidates(
                fta,
                data,
                &mut for_candidates,
                &mut while_candidates,
                &mut fordata_candidates,
                pqs.make_contiguous(),
                q0,
                config.max_big_n_start_index,
            )
        });
    }

    let search_seq_cost =
        start_time.elapsed().as_secs() as f64 + start_time.elapsed().subsec_nanos() as f64 / 1e9;
    time_stats.speculation_cost += search_seq_cost;

    for ForCandidate { q0, l, for_fta } in for_candidates {
        if start_time.elapsed() >= time_left {
            return;
        }
        speculated_transitions.union_inplace(
            synthesize_for(
                fta,
                actions,
                matrices,
                q0,
                NIL_STATE_ID,
                l,
                for_fta,
                time_stats,
            )
            .map(
                |(for_transitions, num_linked_to_middle, num_linked_to_last)| {
                    num_fors_linked_to_middle.fetch_add(num_linked_to_middle, Ordering::SeqCst);
                    num_fors_linked_to_last.fetch_add(num_linked_to_last, Ordering::SeqCst);
                    num_for_candidates.fetch_add(1, Ordering::SeqCst);
                    for_transitions
                },
            )
            .unwrap_or(Transitions::empty()),
        );
    }

    for WhileCandidate { q0, pqs1, ql, ns } in while_candidates {
        if start_time.elapsed() >= time_left {
            return;
        }
        speculated_transitions.union_inplace(
            synthesize_while(
                fta,
                actions,
                matrices,
                q0,
                &pqs1,
                ql,
                ns,
                NIL_STATE_ID,
                time_stats,
            )
            .map(
                |(while_transitions, num_linked_to_middle, num_linked_to_last, num_candidates)| {
                    num_whiles_linked_to_middle.fetch_add(num_linked_to_middle, Ordering::SeqCst);
                    num_whiles_linked_to_last.fetch_add(num_linked_to_last, Ordering::SeqCst);
                    num_while_candidates.fetch_add(num_candidates, Ordering::SeqCst);
                    while_transitions
                },
            )
            .unwrap_or(Transitions::empty()),
        );
    }

    for FordataCandidate {
        q0,
        pqs1,
        ns_map,
        vp,
    } in fordata_candidates
    {
        if start_time.elapsed() >= time_left {
            return;
        }
        speculated_transitions.union_inplace(
            synthesize_fordata(
                fta,
                actions,
                matrices,
                data,
                q0,
                &pqs1,
                ns_map,
                vp,
                NIL_STATE_ID,
                time_stats,
            )
            .map(
                |(
                    fordata_transitions,
                    num_linked_to_middle,
                    num_linked_to_last,
                    num_candidates,
                )| {
                    num_fordatas_linked_to_middle.fetch_add(num_linked_to_middle, Ordering::SeqCst);
                    num_fordatas_linked_to_last.fetch_add(num_linked_to_last, Ordering::SeqCst);
                    num_fordata_candidates.fetch_add(num_candidates, Ordering::SeqCst);
                    fordata_transitions
                },
            )
            .unwrap_or(Transitions::empty()),
        )
    }

    // Merge
    fta.transitions.union_inplace(speculated_transitions);

    let num_for_candidates = num_for_candidates.into_inner();
    if num_for_candidates > 0 {
        println!("Speculated {} Fors", num_for_candidates);
        let num_fors_linked_to_last = num_fors_linked_to_last.into_inner();
        let num_fors_linked_to_middle = num_fors_linked_to_middle.into_inner();
        println!(
            "Synthesized {} Fors ({} linked to last, {} linked to middle)",
            num_fors_linked_to_middle + num_fors_linked_to_last,
            num_fors_linked_to_last,
            num_fors_linked_to_middle,
        );
    }

    let num_while_candidates = num_while_candidates.into_inner();
    if num_while_candidates > 0 {
        println!("Speculated {} Whiles", num_while_candidates);
        let num_whiles_linked_to_last = num_whiles_linked_to_last.into_inner();
        let num_whiles_linked_to_middle = num_whiles_linked_to_middle.into_inner();
        println!(
            "Synthesized {} Whiles ({} linked to last, {} linked to middle)",
            num_whiles_linked_to_middle + num_whiles_linked_to_last,
            num_whiles_linked_to_last,
            num_whiles_linked_to_middle
        );
    }

    let num_fordata_candidates = num_fordata_candidates.into_inner();
    if num_fordata_candidates > 0 {
        println!("Speculated {} ForDatas", num_fordata_candidates);
        let num_fordatas_linked_to_last = num_fordatas_linked_to_last.into_inner();
        let num_fordatas_linked_to_middle = num_fordatas_linked_to_middle.into_inner();
        println!(
            "Synthesized {} ForDatas ({} linked to last, {} linked to middle)",
            num_fordatas_linked_to_middle + num_fordatas_linked_to_last,
            num_fordatas_linked_to_last,
            num_fordatas_linked_to_middle
        );
    }
}

fn synthesize_while(
    fta: &Fta,
    actions: &Actions,
    matrices: &Matrices,
    q0: StateId,
    pqs1: &Vec<(StateId, StateId)>,
    ql: StateId,
    ns: Vector<Selector>,
    qm: StateId,
    time_stats: &mut TimeStats,
) -> Option<(Transitions, usize, usize, usize)> {
    let start_time = Instant::now();
    let mut while_transitions = Transitions::empty();
    let mut num_whiles_linked_to_middle = 0;
    let mut num_whiles_linked_to_last = 0;
    let mut num_ns = 0;

    num_ns += ns.len();
    let (q0_, mut transitions_) = fta.copy_while_body(q0, &pqs1[..pqs1.len() - 1]);
    let q0_annotations = get_state_by_id(q0).annotations;
    assert!(q0_annotations.len() == 1);
    let ql_annotations = get_state_by_id(ql).annotations;
    assert!(ql_annotations.len() == 1);
    let first_iteration_action_trace = ActionTrace(
        q0_annotations[0].output.get_start(),
        ql_annotations[0].output.get_start(),
    );
    let first_iteration_annotations = vec![IOPair {
        input: None, // While always clears the body context
        output: first_iteration_action_trace,
    }];
    let q_while: usize = state_factory::create_new_state_with_annotations(
        GrammarSymbol::Expr,
        first_iteration_annotations,
    );
    transitions_.insert_inplace(Transition::While(q0_, ns.clone()), q_while);
    let while_fta = Fta {
        root: q_while,
        transitions: transitions_,
    };
    let while_depth = while_fta.get_depth(while_fta.root);

    time_stats.speculation_cost +=
        start_time.elapsed().as_secs() as f64 + start_time.elapsed().subsec_nanos() as f64 / 1e9;

    let start_time = Instant::now();
    // output_dot_file(&while_fta, "while_candidate");
    let validate_while_result = validate_candidate_while(
        while_fta,
        q_while,
        q0_,
        q0,
        Some(ql),
        Some(qm),
        &ns,
        actions,
        matrices,
        false,
    );
    println!(
        "synthesize_while {}, (q0={}, l={}, d={}): {}",
        if validate_while_result.is_empty() {
            "failed"
        } else {
            "succeeded"
        },
        q0,
        pqs1.len(),
        while_depth,
        start_time.elapsed().as_millis()
    );
    for result in validate_while_result {
        match result {
            ValidateCandidateLoopResult::LinkedToMiddle(transitions) => {
                num_whiles_linked_to_middle += 1;
                while_transitions.union_inplace(transitions);
            }
            ValidateCandidateLoopResult::LinkedToLast(transitions) => {
                num_whiles_linked_to_last += 1;
                while_transitions.union_inplace(transitions);
            }
        }
    }
    time_stats.validation_cost +=
        start_time.elapsed().as_secs() as f64 + start_time.elapsed().subsec_nanos() as f64 / 1e9;
    if while_transitions.0.is_empty() {
        None
    } else {
        Some((
            while_transitions,
            num_whiles_linked_to_middle,
            num_whiles_linked_to_last,
            num_ns,
        ))
    }
}

pub fn validate_candidate_while(
    while_fta: Fta,
    q_while: StateId,
    q_body: StateId,
    q0: StateId,
    ql: Option<StateId>,
    qm: Option<StateId>,
    ns: &Vector<Selector>,
    actions: &Vec<Action>,
    matrices: &Matrices,
    skip_validate_selector: bool,
) -> Vec<ValidateCandidateLoopResult> {
    let q0_state = state_factory::get_state_by_id(q0);
    assert!(q0_state.annotations.len() == 1);
    let q0_action_trace = q0_state.annotations[0].output;
    let start_action_trace = q0_action_trace;
    let start_dom_trace = start_action_trace.to_start_dom_trace();

    let validate_while_start_time = Instant::now();
    let outputs = ql
        .map(|ql| {
            let ql_state = get_state_by_id(ql);
            let ql_annotations = ql_state.annotations;
            assert!(ql_annotations.len() == 1);
            let second_iteration_start_action_trace = ql_annotations[0].output;
            let second_iteration_start_dom_trace =
                second_iteration_start_action_trace.to_start_dom_trace();
            while_fta.eval_while_skip_first_iteration(
                ActionTrace::empty_at(second_iteration_start_action_trace.get_start()),
                actions,
                matrices,
                second_iteration_start_dom_trace,
                start_dom_trace,
                &None,
                ns,
                q_body,
                q_while,
                skip_validate_selector,
            )
        })
        .unwrap_or_else(|| {
            while_fta.eval_while(
                ActionTrace::empty_at(start_action_trace.get_start()),
                actions,
                matrices,
                start_dom_trace,
                &None,
                ns,
                q_body,
                q_while,
                skip_validate_selector,
            )
        });
    let _validate_while_cost = validate_while_start_time.elapsed().as_secs() as f64
        + validate_while_start_time.elapsed().subsec_nanos() as f64 / 1e9;
    validate_candidate_loop(outputs, q0, qm, q0_action_trace)
}

fn synthesize_fordata(
    fta: &Fta,
    actions: &Actions,
    matrices: &Matrices,
    data: &Option<Value>,
    q0: StateId,
    pqs1: &[(StateId, StateId)],
    ns_map: FxHashMap<StateId, (Vector<Selector>, ValuePath)>,
    vp: ValuePath,
    qm: StateId,
    time_stats: &mut TimeStats,
) -> Option<(Transitions, usize, usize, usize)> {
    let start_time = Instant::now();
    let mut num_loop_linked_to_middle = 0;
    let mut num_loop_linked_to_last = 0;
    let mut loop_transitions = Transitions::empty();

    let mut transitions_ = fta.copy_fordata_body(q0, pqs1, ns_map);
    let q_loop = state_factory::create_new_state(GrammarSymbol::Expr);
    let q0_ = state_factory::copy_state(q0);
    transitions_.insert_inplace(Transition::ForData(vp.clone(), q0_), q_loop);
    let loop_fta = Fta {
        root: q_loop,
        transitions: transitions_,
    };
    time_stats.speculation_cost +=
        start_time.elapsed().as_secs() as f64 + start_time.elapsed().subsec_nanos() as f64 / 1e9;

    let validation_start_time = Instant::now();
    let validate_fordata_result = validate_candidate_fordata(
        loop_fta,
        q_loop,
        q0_,
        q0,
        Some(qm),
        actions,
        matrices,
        data,
        &vp,
    );
    for result in validate_fordata_result {
        match result {
            ValidateCandidateLoopResult::LinkedToMiddle(transitions) => {
                num_loop_linked_to_middle += 1;
                loop_transitions.union_inplace(transitions);
            }
            ValidateCandidateLoopResult::LinkedToLast(transitions) => {
                num_loop_linked_to_last += 1;
                loop_transitions.union_inplace(transitions);
            }
        }
    }
    time_stats.validation_cost += validation_start_time.elapsed().as_secs() as f64
        + validation_start_time.elapsed().subsec_nanos() as f64 / 1e9;
    if loop_transitions.0.is_empty() {
        None
    } else {
        Some((
            loop_transitions,
            num_loop_linked_to_middle,
            num_loop_linked_to_last,
            0,
        ))
    }
}

pub fn validate_candidate_fordata(
    loop_fta: Fta,
    q_loop: StateId,
    q_body: StateId,
    q0: StateId,
    qm: Option<StateId>,
    actions: &Vec<Action>,
    matrices: &Matrices,
    data: &Option<Value>,
    vp: &ValuePath,
) -> Vec<ValidateCandidateLoopResult> {
    let q0_state = state_factory::get_state_by_id(q0);
    assert!(q0_state.annotations.len() == 1);
    let (_, q0_action_trace) = {
        let pair = &q0_state.annotations[0];
        (&pair.input, pair.output)
    };
    let q0_start_dom_trace = q0_action_trace.to_start_dom_trace();

    let outputs = loop_fta.eval_fordata(
        ActionTrace::empty_at(q0_action_trace.get_start()),
        actions,
        matrices,
        data,
        q0_start_dom_trace,
        &None,
        &Some(ValuePath::new()),
        vp,
        q_body,
        q_loop,
    );
    validate_candidate_loop(outputs, q0, qm, q0_action_trace)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionType {
    Nullary(NullaryOp),
    Unary(UnaryOp),
    For,
    ForData,
    While,
    SendKeys,
    SendData,
}

pub fn transition_to_action_type(t: &Transition) -> ActionType {
    match *t {
        Transition::Nullary(op) => ActionType::Nullary(op),
        Transition::Unary(op, _) => ActionType::Unary(op),
        Transition::For(_) => ActionType::For,
        Transition::ForData(_, _) => ActionType::ForData,
        Transition::While(_, _) => ActionType::While,
        Transition::SendKeys(_, _) => ActionType::SendKeys,
        Transition::SendData(_, _) => ActionType::SendData,
        _ => panic!("Expected expression transition but got {}", t),
    }
}

pub fn matches_while_body_pattern(query: &Vec<ActionType>, reference: &Vec<ActionType>) -> bool {
    let query_len = query.len();
    let ref_len = reference.len();
    if query_len < ref_len {
        false
    } else {
        for start in 0..query_len / (ref_len + 1) {
            for offset in 0..ref_len {
                let i = start * (ref_len + 1) + offset;
                if query[i] != reference[offset] {
                    return false;
                }
            }
            // TODO: should be ActionType::Unary(Click), relaxed to Unary for W133 to accommodate scrapeText
            if !matches!(query[start * (ref_len + 1) + ref_len], ActionType::Unary(_)) {
                return false;
            }
        }
        true
    }
}

pub fn get_while_body_pattern_from_fta(fta: &Fta) -> Option<Vec<ActionType>> {
    let mut q = fta.root;
    let mut pattern = vec![];
    let t = fta.transitions.get_by_id(q).unwrap().iter().next().unwrap();
    match *t {
        Transition::While(q_body, _) => {
            q = q_body;
        }
        _ => {
            return None;
        }
    }
    loop {
        if let Some(t) = fta.transitions.get_by_id(q) {
            let t = t.iter().next().unwrap();
            match *t {
                Transition::Nil => break,
                Transition::Seq(p1, q1) => {
                    let t1 = fta
                        .transitions
                        .get_by_id(p1)
                        .unwrap()
                        .iter()
                        .next()
                        .unwrap();
                    // TODO: verify this doesn't disturb While inside another While's body
                    let action_type = transition_to_action_type(t1);
                    if action_type != ActionType::While {
                        pattern.push(action_type);
                    }
                    q = q1;
                }
                _ => panic!("Expecting a Seq or Nil but got {}", t),
            }
        } else {
            return None;
        }
    }
    Some(pattern)
}

pub fn get_while_body_pattern_from_pqs(
    fta: &Fta,
    pqs: &[(StateId, StateId)],
) -> Option<Vec<ActionType>> {
    let mut pattern = vec![];
    for &(p1, _q1) in pqs {
        let t1 = fta
            .transitions
            .get_by_id(p1)
            .unwrap()
            .iter()
            .next()
            .unwrap();
        // TODO: verify this doesn't disturb While inside another While's body
        let action_type = transition_to_action_type(t1);
        pattern.push(action_type);
    }
    Some(pattern)
}

fn is_same_action_type(fta: &Fta, q1: StateId, q2: StateId, has_senddata: &mut bool) -> bool {
    if let Some((is_same_type, has_data)) = get_is_same_action_type(q1, q2) {
        *has_senddata |= has_data;
        return is_same_type;
    }

    let q1_transitions = fta.transitions.get_by_id(q1).unwrap();
    let q2_transitions = fta.transitions.get_by_id(q2).unwrap();

    let has_data = &mut false;
    let is_same_type = {
        let mut result = false;
        'outer: for q1_transition in q1_transitions {
            for q2_transition in q2_transitions {
                result = match (q1_transition, q2_transition) {
                    (&Transition::Seq(q1, q2), &Transition::Seq(q1_, q2_)) => {
                        is_same_action_type(fta, q1, q1_, has_data)
                            && is_same_action_type(fta, q2, q2_, has_data)
                    }
                    (Transition::Nil, Transition::Nil) => true,
                    (Transition::Nullary(op), Transition::Nullary(op_)) => op == op_,
                    (Transition::Unary(op, _), Transition::Unary(op_, _)) => op == op_,
                    (Transition::For(q_body), Transition::For(q_body_)) => {
                        is_same_action_type(fta, *q_body, *q_body_, has_data)
                    }
                    (Transition::ForData(v, q_body), Transition::ForData(v_, q_body_)) => {
                        // TODO: Confirm whether that valuePaths have to be the same
                        v == v_ && is_same_action_type(fta, *q_body, *q_body_, has_data)
                    }
                    (Transition::While(q_body, _), Transition::While(q_body_, _)) => {
                        is_same_action_type(fta, *q_body, *q_body_, has_data)
                    }
                    (Transition::SendKeys(_, str), Transition::SendKeys(_, str_)) => str == str_,
                    (Transition::SendData(_, _), Transition::SendData(_, _)) => {
                        *has_data = true;
                        true
                    }
                    _ => false,
                };
                if result {
                    break 'outer;
                }
            }
        }
        result
    };
    set_is_same_action_type(q1, q2, is_same_type, *has_data);
    *has_senddata |= *has_data;
    is_same_type
}

fn parametrize_for(
    fta: &Fta,
    q0: StateId,
    pqs1: &[(StateId, StateId)],
    pqs2: &[(StateId, StateId)],
    max_big_n_start_index: Index,
) -> Option<Fta> {
    let mut transitions_ = fta.copy_body(q0, pqs1, CopyDepth::ShallowCopy);
    let mut anti_unified = false;
    for (&(p1, _), &(p2, _)) in pqs1.iter().zip(pqs2) {
        let results = parametrize(fta, p1, p2, &mut anti_unified, 0, max_big_n_start_index);
        if results.0.is_empty() {
            return None;
        }
        transitions_.union_inplace(results);
    }

    if !anti_unified {
        None
    } else {
        let q_for = state_factory::create_new_state(GrammarSymbol::Expr);
        let q0_ = state_factory::copy_state(q0);

        transitions_.insert_inplace(Transition::For(q0_), q_for);

        let for_fta = Fta {
            root: q_for,
            transitions: transitions_,
        };
        Some(for_fta)
    }
}

fn parametrize(
    fta: &Fta,
    p1: StateId,
    p2: StateId,
    anti_unified: &mut bool,
    depth: Index,
    max_big_n_start_index: Index,
) -> Transitions {
    let p1_ = copy_state(p1);
    let p1_transitions = fta.transitions.get_by_id(p1).unwrap();
    let p2_transitions = fta.transitions.get_by_id(p2).unwrap();
    let mut transitions_ = Transitions::empty();
    use crate::transition::Transition::*;
    for p1_transition in p1_transitions {
        for p2_transition in p2_transitions {
            match (p1_transition, p2_transition) {
                (Nullary(op1), Nullary(op2)) if op1 == op2 => {
                    transitions_.insert_inplace(p1_transition.clone(), p1_);
                }
                (Unary(op1, ns1), Unary(op2, ns2)) if op1 == op2 => {
                    let (success, ns) = Selector::parametrize(
                        ns1.clone(),
                        ns2.clone(),
                        depth,
                        max_big_n_start_index,
                    );
                    if ns.is_empty() {
                        continue;
                    }
                    *anti_unified |= success;
                    transitions_.insert_inplace(Unary(*op1, ns), p1_);
                }
                (&For(q_body1), &For(q_body2)) => {
                    let mut for_anti_unified = false;
                    let body_transitions_ = parametrize(
                        fta,
                        q_body1,
                        q_body2,
                        &mut for_anti_unified,
                        depth + 1,
                        max_big_n_start_index,
                    );
                    // TODO: may need fix
                    if body_transitions_.0.is_empty() {
                        continue;
                    }
                    *anti_unified |= for_anti_unified;
                    transitions_.union_inplace(body_transitions_);
                    transitions_.insert_inplace(For(copy_state(q_body1)), p1_);
                }
                (Nil, Nil) => {
                    transitions_.insert_inplace(Nil, p1_);
                }
                (&Seq(left1, right1), &Seq(left2, right2)) => {
                    let mut left_anti_unified = false;
                    let left_transitions_ = parametrize(
                        fta,
                        left1,
                        left2,
                        &mut left_anti_unified,
                        depth,
                        max_big_n_start_index,
                    );
                    if left_transitions_.0.is_empty() {
                        continue;
                    }
                    let mut right_anti_unified = false;
                    let right_transitions_ = parametrize(
                        fta,
                        right1,
                        right2,
                        &mut right_anti_unified,
                        depth,
                        max_big_n_start_index,
                    );
                    if right_transitions_.0.is_empty() {
                        continue;
                    }
                    *anti_unified = left_anti_unified | right_anti_unified;
                    transitions_.union_inplace(left_transitions_);
                    transitions_.union_inplace(right_transitions_);
                    transitions_.insert_inplace(Seq(copy_state(left1), copy_state(right1)), p1_);
                }
                (While(q_body1, ns1), While(_q_body2, ns2)) => {
                    // TODO: Check q_body1 is same as q_body2?
                    let (q_body1_, transitions_body_) = fta.deep_copy(*q_body1, true);
                    let (success, ns) =
                        Selector::parametrize(ns1.clone(), ns2.clone(), 0, max_big_n_start_index);
                    if ns.is_empty() {
                        continue;
                    }
                    *anti_unified |= success;
                    transitions_.union_inplace(transitions_body_);
                    transitions_.insert_inplace(While(q_body1_, ns), p1_);
                }
                (SendKeys(ns1, s1), SendKeys(ns2, s2)) if s1 == s2 => {
                    let (success, ns) = Selector::parametrize(
                        ns1.clone(),
                        ns2.clone(),
                        depth,
                        max_big_n_start_index,
                    );
                    if ns.is_empty() {
                        continue;
                    }
                    *anti_unified |= success;
                    transitions_.insert_inplace(SendKeys(ns, s1.clone()), p1_);
                }
                (SendData(_ns1, _v1), SendData(_ns2, _v2)) => {
                    // TODO: Confirm that SendData doesn't need parametrization in For
                    // transitions_.insert_inplace(SendData(ns.clone(), s.clone()), p1_);
                    panic!("SendData is not allowed in For loop");
                }
                _ => {
                    // println!(
                    //     "Impossible parametrize inputs: {} and {}",
                    //     p1_transition,
                    //     p2_transition
                    // )
                    continue;
                }
            }
        }
    }
    transitions_
}

fn synthesize_for(
    fta: &Fta,
    actions: &Actions,
    matrices: &Matrices,
    q0: StateId,
    qm: StateId,
    l: usize,
    for_fta: Fta,
    time_stats: &mut TimeStats,
) -> Option<(Transitions, usize, usize)> {
    let start_time = Instant::now();
    let q0_ = copy_state(q0);
    let for_depth = for_fta.get_depth(for_fta.root);
    let q_for = for_fta.root;

    let validate_for_result = validate_candidate_for(
        fta,
        for_fta,
        q0,
        q_for,
        q0_,
        Some(qm),
        actions,
        matrices,
        true,
    );
    let mut num_linked_to_middle = 0;
    let mut num_linked_to_last = 0;
    println!(
        "synthesize_for {}, (q0={}, l={}, d={}): {}",
        if validate_for_result.is_empty() {
            "failed"
        } else {
            "succeeded"
        },
        q0,
        l,
        for_depth,
        start_time.elapsed().as_millis()
    );
    let rlt =
        Some((
            validate_for_result.into_iter().fold(
                Transitions::empty(),
                |all_transitions, result| match result {
                    ValidateCandidateLoopResult::LinkedToMiddle(transitions) => {
                        num_linked_to_middle += 1;
                        all_transitions.union(transitions)
                    }
                    ValidateCandidateLoopResult::LinkedToLast(transitions) => {
                        num_linked_to_last += 1;
                        all_transitions.union(transitions)
                    }
                },
            ),
            num_linked_to_middle,
            num_linked_to_last,
        ));
    time_stats.validation_cost +=
        start_time.elapsed().as_secs() as f64 + start_time.elapsed().subsec_nanos() as f64 / 1e9;
    rlt
}

#[derive(Debug)]
pub enum ValidateCandidateLoopResult {
    LinkedToLast(Transitions),
    LinkedToMiddle(Transitions),
}

pub fn validate_candidate_for(
    _fta: &Fta,
    for_fta: Fta,
    q0: StateId,
    q_for: StateId,
    q_body: StateId,
    qm: Option<StateId>,
    actions: &Vec<Action>,
    matrices: &Matrices,
    skip_validate_selector: bool,
) -> Vec<ValidateCandidateLoopResult> {
    let q0_state = state_factory::get_state_by_id(q0);
    assert!(q0_state.annotations.len() == 1);
    let (_, q0_action_trace) = {
        let pair = &q0_state.annotations[0];
        (&pair.input, pair.output)
    };
    let q0_start_dom_trace = q0_action_trace.to_start_dom_trace();

    let outputs = for_fta.eval_for(
        ActionTrace::empty_at(q0_action_trace.get_start()),
        actions,
        matrices,
        q0_start_dom_trace,
        &None,
        q_body,
        q_for,
        skip_validate_selector,
    );

    validate_candidate_loop(outputs, q0, qm, q0_action_trace)
}

pub fn validate_candidate_loop(
    outputs: Vec<Fta>,
    q0: StateId,
    qm: Option<StateId>,
    action_trace: ActionTrace,
) -> Vec<ValidateCandidateLoopResult> {
    outputs
        .into_iter()
        .map(|loop_fta| {
            let is_over_max_depth = loop_fta.is_over_max_body_depth(loop_fta.root);
            let (q_loop_action_trace, _) = get_last_output(&loop_fta, 0).unwrap();

            // Add the speculated for transitions
            let mut speculated_transitions = Transitions::empty();

            // Look for a right Seq to connect to
            let start = qm.unwrap_or(q0);
            let pqs_after_qm = Fta::search_top_level_seq(start, NIL_STATE_ID);

            if q_loop_action_trace.get_end() == action_trace.get_end() {
                if let Some(pattern) = get_while_body_pattern_from_fta(&loop_fta) {
                    set_first_while_pattern(q0, pattern)
                }
                speculated_transitions.union_inplace(loop_fta.transitions);
                speculated_transitions
                    .insert_inplace(Transition::Seq(loop_fta.root, NIL_STATE_ID), q0);
                if !is_over_max_depth {
                    add_search_tree_loop_transition(loop_fta.root, NIL_STATE_ID, q0);
                }
                ValidateCandidateLoopResult::LinkedToLast(speculated_transitions)
            } else {
                speculated_transitions.union_inplace(loop_fta.transitions);
                for (pqs, q_start) in &pqs_after_qm {
                    assert_eq!(*q_start, start);
                    for (i, (_, qk)) in pqs.iter().enumerate() {
                        let annotations = &state_factory::get_state_by_id(*qk).annotations;
                        // TODO: Maybe attach Nil annotation to the last Seq?
                        if i == pqs.len() - 1 {
                            assert!(annotations.is_empty());
                            break;
                        }
                        assert!(annotations.len() == 1);
                        let IOPair {
                            output: qk_action_trace,
                            ..
                        } = annotations[0];
                        if q_loop_action_trace.get_end() == qk_action_trace.get_start() {
                            speculated_transitions
                                .insert_inplace(Transition::Seq(loop_fta.root, *qk), q0);
                            if !is_over_max_depth {
                                add_search_tree_loop_transition(loop_fta.root, *qk, q0);
                            }
                        }
                    }
                }
                ValidateCandidateLoopResult::LinkedToMiddle(speculated_transitions)
            }
        })
        .collect()
}
