use super::{action::*, dom::*, env::Env, grammar_symbol::*, io_pair::*, state::StateId};
use crate::dsl::{NullaryOp, UnaryOp};
use crate::env::extend_context;
use crate::fta::Fta;
use crate::selector::{Index, Selector};
use crate::selector_matrix::{validate_selector, Matrices};
use crate::state_factory::{
    create_new_state_with_annotations, get_annotated_new_state, get_for_body_last_stopped_dom,
    get_is_early_stopped, get_last_output, get_state_by_id, get_while_body_last_stopped_dom,
    set_for_body_last_stopped_dom, set_is_early_stopped, set_while_body_last_stopped_dom,
};
use crate::transition::{Transition, Transitions};
use crate::value_path::{self, get_list_at_path, ValuePath};
use im_rc::{vector, Vector};
use itertools::Itertools;
use serde_json::Value;
use smallvec::{smallvec, SmallVec};

/// Successful result of evaluating a While
#[derive(Clone)]
pub struct EvalWhileResult {
    fta_i_1: Fta,
    fta: Fta,
    ns: Vector<Selector>,
    clicked: bool,
}

impl Fta {
    /// Rules (2~5)
    fn eval(
        &self,
        action_trace: ActionTrace,
        actions: &Actions,
        matrices: &Matrices,
        data: &Option<Value>,
        dom_trace: DomTrace,
        context: &Env,
        data_context: &Option<ValuePath>,
        q: StateId,
        skip_validate_selector: bool,
    ) -> Vec<Fta> {
        use Transition::*;
        if dom_trace.is_empty() {
            set_is_early_stopped(true);
            // Rule (2)
            vec![Fta {
                root: q,
                transitions: Transitions::empty(),
            }]
        } else {
            let input_alphabets = self.transitions.get_by_id(q).unwrap_or_else(|| {
                panic!("Must have a transition to the input state with id {}.", q)
            });
            input_alphabets
                .iter()
                .flat_map(|input_alphabet| match input_alphabet {
                    // Rule (3)
                    &Seq(q1, q2) => self.eval_seq(
                        action_trace,
                        actions,
                        matrices,
                        data,
                        dom_trace,
                        context,
                        data_context,
                        q1,
                        q2,
                        q,
                        skip_validate_selector,
                    ),
                    &Nullary(op) => {
                        self.eval_nullary(action_trace, actions, dom_trace, context, q, op)
                    }
                    // Rule (4) and (5)
                    Unary(op, ns) => self.eval_unary(
                        action_trace,
                        actions,
                        matrices,
                        dom_trace,
                        context,
                        q,
                        *op,
                        ns,
                        skip_validate_selector,
                    ),
                    For(q3) => self.eval_for(
                        action_trace,
                        actions,
                        matrices,
                        dom_trace,
                        context,
                        *q3,
                        q,
                        skip_validate_selector,
                    ),
                    While(q1, ns) => self.eval_while(
                        action_trace,
                        actions,
                        matrices,
                        dom_trace,
                        context,
                        ns,
                        *q1,
                        q,
                        skip_validate_selector,
                    ),
                    SendKeys(ns, s) => self.eval_sendkeys(
                        action_trace,
                        actions,
                        matrices,
                        dom_trace,
                        context,
                        q,
                        ns,
                        s,
                        skip_validate_selector,
                    ),
                    SendData(ns, v) => self.eval_senddata(
                        action_trace,
                        actions,
                        matrices,
                        data,
                        dom_trace,
                        context,
                        data_context,
                        q,
                        ns,
                        v,
                    ),
                    ForData(v, q3) => self.eval_fordata(
                        action_trace,
                        actions,
                        matrices,
                        data,
                        dom_trace,
                        context,
                        data_context,
                        v,
                        *q3,
                        q,
                    ),
                    Nil => {
                        // similar to Rule (2)
                        let mut fta_ = Fta {
                            root: q,
                            transitions: Transitions::empty(),
                        };
                        fta_.add_transition(Transition::Nil, q);
                        vec![fta_]
                    }
                })
                .collect()
        }
    }

    /// Rule (3)
    fn eval_seq(
        &self,
        action_trace: ActionTrace,
        actions: &Actions,
        matrices: &Matrices,
        data: &Option<Value>,
        dom_trace: DomTrace,
        context: &Env,
        data_context: &Option<ValuePath>,
        q1: StateId,
        q2: StateId,
        q: StateId,
        skip_validate_selector: bool,
    ) -> Vec<Fta> {
        let results1 = self.eval(
            action_trace,
            actions,
            matrices,
            data,
            dom_trace,
            context,
            data_context,
            q1,
            skip_validate_selector,
        );
        results1
            .into_iter()
            .flat_map(|fta1| {
                let (action_trace1, dom_trace1) =
                    get_last_output(&fta1, dom_trace.number_of_doms).unwrap();
                let results2 = self.eval(
                    action_trace1,
                    actions,
                    matrices,
                    data,
                    dom_trace1,
                    context,
                    data_context,
                    q2,
                    skip_validate_selector,
                );
                results2
                    .iter()
                    .map(|fta2| {
                        let (action_trace2, _) = get_last_output(fta2, dom_trace.number_of_doms)
                            .unwrap_or((action_trace1.emptied(), dom_trace1));
                        // println!("eval_seq action_trace2: {:?}", action_trace2);
                        let annotation = IOPair {
                            input: context.clone(),
                            output: action_trace1.merge(&action_trace2),
                        };
                        let q_ = get_annotated_new_state(GrammarSymbol::Program, q, annotation);
                        // println!("validate seq annotation of state {:?} to {:?}", q_, get_state_by_id(q_).annotations);

                        let mut fta3 = Fta {
                            root: q_,
                            transitions: fta1.transitions.clone().union(fta2.transitions.clone()),
                        };
                        fta3.add_transition(Transition::Seq(fta1.root, fta2.root), q_);
                        fta3
                    })
                    .group_by(|fta| fta.root)
                    .into_iter()
                    .map(|(root, ftas)| Fta {
                        root,
                        transitions: ftas.fold(Transitions::empty(), |transitions, fta| {
                            transitions.union(fta.transitions)
                        }),
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn eval_nullary(
        &self,
        action_trace: ActionTrace,
        actions: &Actions,
        dom_trace: DomTrace,
        context: &Env,
        q: StateId,
        op: NullaryOp,
    ) -> Vec<Fta> {
        if !action_trace.is_consistent_with(&Action::Nullary(op), actions) {
            vec![]
        } else {
            let action_trace_ = action_trace.next();
            let _dom_trace_ = dom_trace.next();
            let annotation = IOPair {
                input: context.clone(),
                output: action_trace_,
            };
            let q_ = get_annotated_new_state(GrammarSymbol::Expr, q, annotation);
            let mut fta_ = Fta {
                root: q_,
                transitions: Transitions::empty(),
            };
            fta_.add_transition(Transition::Nullary(op), q_);
            vec![fta_]
        }
    }

    /// Rule (4) and (5)
    fn eval_unary(
        &self,
        action_trace: ActionTrace,
        actions: &Actions,
        matrices: &Matrices,
        dom_trace: DomTrace,
        context: &Env,
        q: StateId,
        op: UnaryOp,
        ns: &Vector<Selector>,
        skip_validate_selector: bool,
    ) -> Vec<Fta> {
        assert!(!dom_trace.is_empty());

        let matrix_index = dom_trace.n;

        // Rule (4)
        if !action_trace.is_consistent_with(&Action::Unary(matrix_index, op), actions) {
            vec![]
        } else if skip_validate_selector {
            let action_trace_ = action_trace.next();
            let annotation = IOPair {
                input: context.clone(),
                output: action_trace_,
            };
            let q_ = get_annotated_new_state(GrammarSymbol::Expr, q, annotation);
            let mut fta_ = Fta {
                root: q_,
                transitions: Transitions::empty(),
            };
            fta_.add_transition(Transition::Unary(op, ns.clone()), q_);
            vec![fta_]
        } else {
            let mut valid_ns: SmallVec<[_; 4]> = smallvec![];
            for n in ns.iter() {
                let rho = n.to_concrete_selector(context);
                if validate_selector(matrices, matrix_index, &rho) {
                    valid_ns.push(n);
                }
            }

            if valid_ns.is_empty() {
                vec![]
            } else {
                let action_trace_ = action_trace.next();
                let annotation = IOPair {
                    input: context.clone(),
                    output: action_trace_,
                };
                let q_ = get_annotated_new_state(GrammarSymbol::Expr, q, annotation);
                let mut fta_ = Fta {
                    root: q_,
                    transitions: Transitions::empty(),
                };

                if valid_ns.len() == ns.len() {
                    // fast path: copying ns is O(1)
                    fta_.add_transition(Transition::Unary(op, ns.clone()), q_);
                } else {
                    // slow path: cloning
                    let valid_ns = valid_ns.into_iter().cloned().collect();
                    fta_.add_transition(Transition::Unary(op, valid_ns), q_);
                }
                vec![fta_]
            }
        }
    }

    /// Used by Rule (16)
    pub fn eval_click_expr(
        action_trace: ActionTrace,
        actions: &Actions,
        dom_trace: DomTrace,
        rho: Index,
    ) -> Result<IOPairOutput, ()> {
        if action_trace.is_consistent_with(&Action::Unary(rho, UnaryOp::Click), actions) {
            Ok((action_trace.next(), dom_trace.next()))
        } else {
            Err(())
        }
    }

    /// Rule (7)
    pub fn eval_for(
        &self,
        action_trace: ActionTrace,
        actions: &Actions,
        matrices: &Matrices,
        dom_trace: DomTrace,
        context: &Env,
        q3: StateId,
        q: StateId,
        skip_validate_selector: bool,
    ) -> Vec<Fta> {
        // output_dot_file(self, &q.to_string());
        let results3 = Fta {
            root: q3,
            transitions: self.transitions.clone(),
        }
        .eval_for_helper(
            Fta {
                root: q3,
                transitions: Transitions::empty(),
            },
            action_trace,
            actions,
            matrices,
            dom_trace,
            context,
            q3,
            1,
            skip_validate_selector,
        );

        results3
            .into_iter()
            .filter_map(|(fta3_i_1, mut fta3)| {
                let fta3_annotations = get_state_by_id(fta3.root).annotations;

                // TODO: fta3_annotations shouldn't be empty? Why is it empty for W110T2
                if fta3_annotations.is_empty()
                    || fta3_annotations.last().unwrap().output.1 <= dom_trace.n
                {
                    return None;
                }

                let action_trace3_ =
                    ActionTrace(dom_trace.n, fta3_annotations.last().unwrap().output.1);
                let annotation = IOPair {
                    input: context.clone(),
                    output: action_trace3_,
                };
                let q_ = get_annotated_new_state(GrammarSymbol::Expr, q, annotation);
                fta3.clean_up_insert_missing(fta3_i_1);
                fta3.add_transition(Transition::For(fta3.root), q_);
                fta3.root = q_;
                Some(fta3)
            })
            .group_by(|fta| fta.root)
            .into_iter()
            .map(|(root, ftas)| Fta {
                root,
                transitions: ftas.fold(Transitions::empty(), |transitions, fta| {
                    transitions.union(fta.transitions)
                }),
            })
            .collect::<Vec<_>>()
    }

    /// Rule (10-12)
    fn eval_for_helper(
        self,
        fta_i_1: Fta,
        action_trace: ActionTrace,
        actions: &Actions,
        matrices: &Matrices,
        dom_trace: DomTrace,
        context: &Env,
        q: StateId,
        i: Index,
        skip_validate_selector: bool,
    ) -> Vec<(Fta, Fta)> {
        // Rule (12)
        if dom_trace.is_empty() {
            if !get_is_early_stopped() {
                set_for_body_last_stopped_dom(q, dom_trace.n);
                set_is_early_stopped(false);
            }
            return vec![(fta_i_1, self)];
        }

        let mut action_trace = action_trace;
        let mut dom_trace = dom_trace;
        let mut i = i;

        let annotations = get_state_by_id(q).annotations;
        let mut is_partial_last_iteration = false;
        if i == 1 {
            match get_for_body_last_stopped_dom(q) {
                Some(n) if n >= dom_trace.n => {
                    let (rev_annotation_index, _) = annotations
                        .iter()
                        .rev()
                        .find_position(|annotation| annotation.output.0 == dom_trace.n)
                        .unwrap();
                    action_trace = ActionTrace(dom_trace.n, n);
                    dom_trace = DomTrace {
                        n,
                        number_of_doms: dom_trace.number_of_doms,
                    };
                    i = rev_annotation_index as Index + 2;
                }
                _ => {
                    if annotations.len() >= 2
                        && annotations[annotations.len() - 2].output.0 >= dom_trace.n
                    {
                        if let Some((rev_annotation_index, _)) = annotations
                            .iter()
                            .rev()
                            .find_position(|annotation| annotation.output.0 == dom_trace.n)
                        {
                            action_trace = ActionTrace(
                                dom_trace.n,
                                annotations[annotations.len() - 2].output.1,
                            );
                            dom_trace = DomTrace {
                                n: annotations[annotations.len() - 2].output.1,
                                number_of_doms: dom_trace.number_of_doms,
                            };
                            i = rev_annotation_index as Index + 1;
                            is_partial_last_iteration = true;
                        }
                    }
                }
            }
        }

        // push n[i] selector into context
        let body_context = extend_context(context, i);
        let is_valid = self
            .validate(action_trace, actions, matrices, dom_trace, &body_context, q)
            .is_some();
        if is_valid {
            let results1 = self.eval(
                action_trace,
                actions,
                matrices,
                &None,
                dom_trace,
                &body_context,
                &None,
                q,
                skip_validate_selector,
            );
            return results1
                .into_iter()
                .flat_map(|fta1| {
                    let (action_trace1, dom_trace1) =
                        get_last_output(&fta1, dom_trace.number_of_doms)
                            .unwrap_or((action_trace.emptied(), dom_trace));
                    // TODO: repeated evaluation across iterations?
                    let q_body = fta1.root;
                    fta1.eval_for_helper(
                        self.clone(),
                        action_trace1,
                        actions,
                        matrices,
                        dom_trace1,
                        context,
                        q_body,
                        i + 1,
                        skip_validate_selector,
                    )
                })
                .collect();
        } else if is_partial_last_iteration {
            // last iteration ran partially in last incremental
            // Now, however, validation fails with one more action
            let mut root_annotations = get_state_by_id(self.root).annotations;
            if root_annotations.last().unwrap().output.1 > action_trace.1 {
                // println!("wrong partial last iteration");
                root_annotations.pop();
                let new_root =
                    create_new_state_with_annotations(GrammarSymbol::Program, root_annotations);
                let mut result_fta = Fta {
                    root: new_root,
                    transitions: self.transitions.clone(),
                };
                // TODO: Confirm that we don't need to also delete extra annotations from last partial iteration
                for relink_transition in self.transitions.get_by_id(self.root).unwrap() {
                    result_fta
                        .transitions
                        .remove_transition(self.root, relink_transition);
                    result_fta.add_transition(relink_transition.clone(), new_root);
                }
                return vec![(fta_i_1, result_fta)];
            }
        }
        vec![(fta_i_1, self)]
    }

    /// Rules (13) to (16)
    pub fn eval_while_skip_first_iteration(
        &self,
        second_iteration_action_trace: ActionTrace,
        actions: &Actions,
        matrices: &Matrices,
        second_iteration_dom_trace: DomTrace,
        first_iteration_dom_trace: DomTrace,
        context: &Env,
        ns: &Vector<Selector>,
        q_body: StateId,
        q: StateId,
        skip_validate_selector: bool,
    ) -> Vec<Fta> {
        let results = self.eval_while_helper(
            second_iteration_action_trace,
            actions,
            matrices,
            second_iteration_dom_trace,
            context,
            ns,
            q_body,
            skip_validate_selector,
        );
        results
            .into_iter()
            .filter_map(|result| {
                let EvalWhileResult {
                    fta_i_1,
                    fta: fta_,
                    ns: ns_,
                    clicked,
                } = result;
                let fta_annotations = get_state_by_id(fta_.root).annotations;
                if fta_annotations.len() < 2 {
                    return None;
                }
                let action_trace_ = ActionTrace(
                    first_iteration_dom_trace.n,
                    fta_annotations.last().unwrap().output.1 + if clicked { 1 } else { 0 },
                );
                let annotation = IOPair {
                    input: None, // While always clears the body context
                    output: action_trace_,
                };
                let q_ = get_annotated_new_state(GrammarSymbol::Expr, q, annotation);
                let mut result_fta = Fta {
                    root: q_,
                    transitions: fta_.transitions.union(fta_i_1.transitions),
                };
                result_fta.add_transition(Transition::While(fta_.root, ns_), q_);
                Some(result_fta)
            })
            .group_by(|fta| fta.root)
            .into_iter()
            .map(|(root, ftas)| Fta {
                root,
                transitions: ftas.fold(Transitions::empty(), |transitions, fta| {
                    transitions.union(fta.transitions)
                }),
            })
            .collect()
    }

    /// Rules (13) to (16)
    pub fn eval_while(
        &self,
        action_trace: ActionTrace,
        actions: &Actions,
        matrices: &Matrices,
        dom_trace: DomTrace,
        context: &Env,
        ns: &Vector<Selector>,
        q_body: StateId,
        q: StateId,
        skip_validate_selector: bool,
    ) -> Vec<Fta> {
        let results = self.eval_while_helper(
            action_trace,
            actions,
            matrices,
            dom_trace,
            context,
            ns,
            q_body,
            skip_validate_selector,
        );
        results
            .into_iter()
            .map(|result| {
                let EvalWhileResult {
                    fta_i_1,
                    fta: fta_,
                    ns: ns_,
                    clicked,
                } = result;
                let fta_annotations = get_state_by_id(fta_.root).annotations;
                let action_trace_ = ActionTrace(
                    dom_trace.n,
                    fta_annotations.last().unwrap().output.1 + if clicked { 1 } else { 0 },
                );
                let annotation = IOPair {
                    input: None, // While always clears the body context
                    output: action_trace_,
                };
                let q_ = get_annotated_new_state(GrammarSymbol::Expr, q, annotation);
                let mut result_fta = Fta {
                    root: q_,
                    transitions: fta_.transitions.union(fta_i_1.transitions),
                };
                result_fta.add_transition(Transition::While(fta_.root, ns_), q_);
                result_fta
            })
            .group_by(|fta| fta.root)
            .into_iter()
            .map(|(root, ftas)| Fta {
                root,
                transitions: ftas.fold(Transitions::empty(), |transitions, fta| {
                    transitions.union(fta.transitions)
                }),
            })
            .collect()
    }

    fn eval_while_helper(
        &self,
        action_trace: ActionTrace,
        actions: &Actions,
        matrices: &Matrices,
        dom_trace: DomTrace,
        context: &Env,
        ns: &Vector<Selector>,
        q_body: StateId,
        skip_validate_selector: bool,
    ) -> Vec<EvalWhileResult> {
        let mut action_trace = action_trace;
        let mut dom_trace = dom_trace;
        let empty_context = None;

        match get_while_body_last_stopped_dom(q_body) {
            Some(n) if n >= dom_trace.n => {
                action_trace = ActionTrace(dom_trace.n, n);
                dom_trace = DomTrace {
                    n,
                    number_of_doms: dom_trace.number_of_doms,
                };
            }
            _ => {
                let annotations = get_state_by_id(q_body).annotations;
                if annotations.len() >= 2
                    && annotations[annotations.len() - 2].output.0 >= dom_trace.n
                    && annotations
                        .iter()
                        .rev()
                        .any(|annotation| annotation.output.0 == dom_trace.n)
                {
                    // First While incremental after synthesis
                    // always skip the first 2 iterations because they are full iterations
                    if annotations.len() == 2 {
                        // println!("Skipped {} While iterations", 2);
                        action_trace = ActionTrace(
                            dom_trace.n,
                            annotations[annotations.len() - 1].output.1 + 1,
                        );
                        dom_trace = DomTrace {
                            n: annotations[annotations.len() - 1].output.1 + 1,
                            number_of_doms: dom_trace.number_of_doms,
                        };
                    } else {
                        action_trace = ActionTrace(
                            dom_trace.n,
                            annotations[annotations.len() - 2].output.1 + 1,
                        );
                        dom_trace = DomTrace {
                            n: annotations[annotations.len() - 2].output.1 + 1,
                            number_of_doms: dom_trace.number_of_doms,
                        };
                    }
                }
            }
        };

        let mut valid = false;
        let first_seq = self
            .transitions
            .get_by_id(q_body)
            .unwrap()
            .iter()
            .next()
            .unwrap();
        match first_seq {
            Transition::Seq(left, _right) => {
                valid = self
                    .validate(
                        action_trace,
                        actions,
                        matrices,
                        dom_trace,
                        &empty_context,
                        *left,
                    )
                    .is_some();
            }
            _ => panic!("eval_while_helper: transition to q_body is not a Seq"),
        }
        if valid {
            let results1 = self.eval(
                action_trace,
                actions,
                matrices,
                &None,
                dom_trace,
                &empty_context,
                &None,
                q_body,
                skip_validate_selector,
            );
            results1
                .into_iter()
                .flat_map(|fta1| {
                    let (action_trace1_, dom_trace1_) =
                        get_last_output(&fta1, dom_trace.number_of_doms)
                            .unwrap_or((action_trace.emptied(), dom_trace));
                    // Rule (14)
                    if dom_trace1_.is_empty() {
                        vec![EvalWhileResult {
                            fta_i_1: self.clone(),
                            fta: fta1,
                            ns: ns.clone(),
                            clicked: false,
                        }]
                    }
                    // Rule (15) and (16)
                    else {
                        let matrix_index = dom_trace1_.n;
                        let mut valid_ns = vector![];
                        let mut invalid_ns = vector![];
                        for n in ns {
                            let rho = n.to_concrete_selector(context);
                            if validate_selector(matrices, matrix_index, &rho) {
                                valid_ns.push_back(n.clone());
                            } else {
                                invalid_ns.push_back(n.clone());
                            }
                        }
                        if valid_ns.is_empty() {
                            // Rule (15)
                            // TODO: Verify that we can just return q2 instead of q2_
                            // when we remove Bottom as a variant in selectors
                            vec![EvalWhileResult {
                                fta_i_1: self.clone(),
                                fta: fta1,
                                ns: invalid_ns,
                                clicked: false,
                            }]
                        } else {
                            // Rule (16)
                            match Self::eval_click_expr(
                                action_trace1_,
                                actions,
                                dom_trace1_,
                                matrix_index,
                            ) {
                                Ok((action_trace2_, dom_trace2_)) => {
                                    if dom_trace2_.is_empty() {
                                        set_while_body_last_stopped_dom(fta1.root, dom_trace2_.n);
                                        return vec![EvalWhileResult {
                                            fta_i_1: self.clone(),
                                            fta: fta1,
                                            ns: valid_ns.clone(),
                                            clicked: true,
                                        }];
                                    }
                                    let mut results3 = fta1.eval_while_helper(
                                        action_trace2_,
                                        actions,
                                        matrices,
                                        dom_trace2_,
                                        context,
                                        &valid_ns,
                                        fta1.root,
                                        skip_validate_selector,
                                    );
                                    // invalid_ns stop at this iteration
                                    // Create an annotated While containing those invalid_ns
                                    if !invalid_ns.is_empty() {
                                        results3.push(EvalWhileResult {
                                            fta_i_1: self.clone(),
                                            fta: fta1,
                                            ns: invalid_ns,
                                            clicked: false,
                                        });
                                    }
                                    results3
                                }
                                Err(_) => vec![EvalWhileResult {
                                    fta_i_1: self.clone(),
                                    fta: fta1,
                                    ns: invalid_ns,
                                    clicked: false,
                                }],
                            }
                        }
                    }
                })
                .collect()
        } else {
            // println!("root: {}", self;
            vec![EvalWhileResult {
                fta_i_1: self.clone(),
                fta: Fta {
                    root: q_body,
                    transitions: self.transitions.clone(),
                },
                ns: ns.clone(),
                clicked: true,
            }]
        }
    }

    fn eval_sendkeys(
        &self,
        action_trace: ActionTrace,
        actions: &Actions,
        matrices: &Matrices,
        dom_trace: DomTrace,
        context: &Env,
        q: StateId,
        ns: &Vector<Selector>,
        s: &String,
        skip_validate_selector: bool,
    ) -> Vec<Fta> {
        assert!(!dom_trace.is_empty());

        let matrix_index = dom_trace.n;

        // Rule (4)
        if !action_trace.is_consistent_with(&Action::SendKeys(matrix_index, s.clone()), actions) {
            vec![]
        } else if skip_validate_selector {
            let action_trace_ = action_trace.next();
            let annotation = IOPair {
                input: context.clone(),
                output: action_trace_,
            };
            let q_ = get_annotated_new_state(GrammarSymbol::Expr, q, annotation);
            let mut fta_ = Fta {
                root: q_,
                transitions: Transitions::empty(),
            };

            fta_.add_transition(Transition::SendKeys(ns.clone(), s.clone()), q_);

            vec![fta_]
        } else {
            let mut valid_ns: SmallVec<[_; 4]> = smallvec![];
            for n in ns.iter() {
                let rho = n.to_concrete_selector(context);
                if validate_selector(matrices, matrix_index, &rho) {
                    valid_ns.push(n);
                }
            }

            if valid_ns.is_empty() {
                vec![]
            } else {
                let action_trace_ = action_trace.next();
                let annotation = IOPair {
                    input: context.clone(),
                    output: action_trace_,
                };
                let q_ = get_annotated_new_state(GrammarSymbol::Expr, q, annotation);
                let mut fta_ = Fta {
                    root: q_,
                    transitions: Transitions::empty(),
                };

                if valid_ns.len() == ns.len() {
                    // fast path: copying ns is O(1)
                    fta_.add_transition(Transition::SendKeys(ns.clone(), s.clone()), q_);
                } else {
                    // slow path: cloning
                    let valid_ns = valid_ns.into_iter().cloned().collect();
                    fta_.add_transition(Transition::SendKeys(valid_ns, s.clone()), q_);
                }
                vec![fta_]
            }
        }
    }

    fn eval_senddata(
        &self,
        action_trace: ActionTrace,
        actions: &Actions,
        matrices: &Matrices,
        data: &Option<Value>,
        dom_trace: DomTrace,
        context: &Env,
        data_context: &Option<ValuePath>,
        q: StateId,
        ns: &Vector<Selector>,
        v: &ValuePath,
    ) -> Vec<Fta> {
        if data_context.is_none() || data.is_none() {
            return vec![];
        }
        assert!(!dom_trace.is_empty());
        let matrix_index = dom_trace.n;
        // find any suffix
        let data_context = data_context.as_ref().unwrap();
        let suffix = if data_context.len() >= v.0.len() {
            ValuePath::new()
        } else {
            ValuePath::from_path(v.0[data_context.len()..].to_vec())
        };
        let concrete_vp = data_context.clone().append(&suffix);

        // Rule (4)
        if !action_trace.is_consistent_with(&Action::SendData(matrix_index, concrete_vp), actions) {
            vec![]
        } else {
            let mut valid_ns: SmallVec<[_; 4]> = smallvec![];
            for n in ns.iter() {
                let rho = n.to_concrete_selector(context);
                if validate_selector(matrices, matrix_index, &rho) {
                    valid_ns.push(n);
                }
            }
            if valid_ns.is_empty() {
                vec![]
            } else {
                let action_trace_ = action_trace.next();
                let annotation = IOPair {
                    input: context.clone(),
                    output: action_trace_,
                };
                let q_ = get_annotated_new_state(GrammarSymbol::Expr, q, annotation);
                let mut fta_ = Fta {
                    root: q_,
                    transitions: Transitions::empty(),
                };
                if valid_ns.len() == ns.len() {
                    // fast path: copying ns is O(1)
                    fta_.add_transition(Transition::SendData(ns.clone(), v.clone()), q_);
                } else {
                    // slow path: cloning
                    let valid_ns = valid_ns.into_iter().cloned().collect();
                    fta_.add_transition(Transition::SendData(valid_ns, v.clone()), q_);
                }
                vec![fta_]
            }
        }
    }

    pub fn eval_fordata(
        &self,
        action_trace: ActionTrace,
        actions: &Actions,
        matrices: &Matrices,
        data: &Option<Value>,
        dom_trace: DomTrace,
        context: &Env,
        data_context: &Option<ValuePath>,
        v: &ValuePath,
        q_body: StateId,
        q: StateId,
    ) -> Vec<Fta> {
        if data_context.is_none() || data.is_none() {
            return vec![];
        }
        let data_ = data.as_ref().unwrap();
        let data_context_ = data_context.as_ref().unwrap();
        let body_context = data_context_.append(v);
        let data_len = if let Some(value_list) = get_list_at_path(data_, &body_context.0) {
            value_list.len()
        } else {
            0
        };
        if data_len == 0 {
            return vec![];
        }

        let results = self.eval_fordata_helper(
            Fta {
                root: q_body,
                transitions: Transitions::empty(),
            },
            action_trace,
            actions,
            matrices,
            data,
            dom_trace,
            context,
            &Some(body_context),
            q_body,
            0,
            data_len,
        );
        results
            .into_iter()
            .filter_map(|(result_i_1, result)| {
                let result_annotations = get_state_by_id(result.root).annotations;

                if result_annotations.is_empty()
                    || result_annotations.last().unwrap().output.1 <= dom_trace.n
                {
                    return None;
                }

                let action_trace_ =
                    ActionTrace(dom_trace.n, result_annotations.last().unwrap().output.1);
                let annotation = IOPair {
                    input: context.clone(),
                    output: action_trace_,
                };
                let q_ = get_annotated_new_state(GrammarSymbol::Expr, q, annotation);
                let mut fta_ = Fta {
                    root: q_,
                    transitions: result.transitions.union(result_i_1.transitions),
                };
                // fta_.clean_up(fta_.root);
                fta_.add_transition(Transition::ForData(v.clone(), result.root), q_);
                Some(fta_)
            })
            .group_by(|fta| fta.root)
            .into_iter()
            .map(|(root, ftas)| Fta {
                root,
                transitions: ftas.fold(Transitions::empty(), |transitions, fta| {
                    transitions.union(fta.transitions)
                }),
            })
            .collect::<Vec<_>>()
    }

    fn eval_fordata_helper(
        &self,
        fta_i_1: Fta,
        action_trace: ActionTrace,
        actions: &Actions,
        matrices: &Matrices,
        data: &Option<Value>,
        dom_trace: DomTrace,
        context: &Env,
        data_context: &Option<ValuePath>,
        q: StateId,
        i: usize,
        data_len: usize,
    ) -> Vec<(Fta, Fta)> {
        if dom_trace.is_empty() {
            return vec![(fta_i_1, self.clone())];
        }

        let mut action_trace = action_trace;
        let mut dom_trace = dom_trace;
        let mut i = i;

        let annotations = get_state_by_id(q).annotations;
        if i == 0
            && annotations.len() >= 2
            && annotations[annotations.len() - 2].output.0 >= dom_trace.n
        {
            if let Some((rev_annotation_index, _)) = annotations
                .iter()
                .rev()
                .find_position(|annotation| annotation.output.0 == dom_trace.n)
            {
                action_trace =
                    ActionTrace(dom_trace.n, annotations[annotations.len() - 2].output.1);
                dom_trace = DomTrace {
                    n: annotations[annotations.len() - 2].output.1,
                    number_of_doms: dom_trace.number_of_doms,
                };
                i = rev_annotation_index;
            }
        }

        // push n[i] valuepath into context
        let body_context = data_context
            .as_ref()
            .unwrap()
            .append_index(value_path::Index::Number(i));
        if i < data_len {
            let results1 = self.eval(
                action_trace,
                actions,
                matrices,
                data,
                dom_trace,
                context,
                &Some(body_context),
                q,
                false,
            );
            results1
                .into_iter()
                .flat_map(|fta1| {
                    let (action_trace1, dom_trace1) =
                        get_last_output(&fta1, dom_trace.number_of_doms)
                            .unwrap_or((action_trace.emptied(), dom_trace));
                    fta1.eval_fordata_helper(
                        self.clone(),
                        action_trace1,
                        actions,
                        matrices,
                        data,
                        dom_trace1,
                        context,
                        data_context,
                        fta1.root,
                        i + 1,
                        data_len,
                    )
                })
                .collect()
        } else {
            vec![(fta_i_1, self.clone())]
        }
    }
}
