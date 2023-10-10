use std::time::{Duration, Instant};

use serde_json::Value;

use crate::{
    action::{Action, ActionTrace, Actions},
    dom::DomTrace,
    dsl::program_to_string,
    fta::Fta,
    grammar_symbol::GrammarSymbol,
    io_pair::IOPair,
    selector_matrix::{ConcreteSelectorMatrix, Matrices},
    state::NIL_STATE_ID,
    state_factory::{
        self, create_new_state, create_new_state_with_annotations,
        remove_search_tree_loop_transition, set_search_tree_top_level_transition,
    },
    synthesis::{
        validate_candidate_for, validate_candidate_fordata, validate_candidate_while,
        ValidateCandidateLoopResult,
    },
    test_utils::BenchmarkConfig,
    transition::{Transition, Transitions},
};

impl Fta {
    pub fn add_next_action(
        &mut self,
        action: Action,
        matrix: &ConcreteSelectorMatrix,
        data: &Option<Value>,
        actions: &Actions,
        matrices: &Matrices,
        dom_trace: DomTrace,
        config: &BenchmarkConfig,
        num_selectors: Option<usize>,
        time_left: Duration,
    ) -> Option<bool> {
        let start_time = Instant::now();

        // find the state that is the top level seq over nil

        // Find all Fors linked to the final Nil and the action_trace of the top-level seq over Fors
        let mut fors_to_copy = vec![];
        let mut whiles_to_copy = vec![];
        let mut fordatas_to_copy = vec![];
        let mut transitions_to_relink = vec![];
        let mut curr_top_level_state = self.root;
        loop {
            let top_level_transitions = self.transitions.get_by_id(curr_top_level_state).unwrap();
            let old_top_level_state = curr_top_level_state;
            let mut next_top_level_state = curr_top_level_state;
            for top_level_transition in top_level_transitions {
                match top_level_transition {
                    // Reached the end of top-level states
                    Transition::Nil => {
                        // Should only contain a single Nil transition
                        assert!(top_level_transitions.len() == 1);
                    }
                    // More top-level states to go
                    &Transition::Seq(left_state, right_state) => {
                        let transitions = match self.transitions.get_by_id(left_state) {
                            Some(transitions) => transitions,
                            None => {
                                panic!("State with id {} has no input transitions.", left_state)
                            }
                        };
                        let mut found_loop = false;
                        for transition in transitions {
                            match transition {
                                Transition::For(q_body) => {
                                    if right_state == NIL_STATE_ID {
                                        // TODO: if q_for_action_trace.get_end() == action_trace.get_end() {
                                        // TODO: change the annotation of q0
                                        let q0_annotations =
                                            state_factory::get_state_by_id(curr_top_level_state)
                                                .annotations;
                                        assert!(q0_annotations.len() == 1);
                                        fors_to_copy.push((
                                            curr_top_level_state,
                                            *q_body,
                                            left_state,
                                        ));
                                    }
                                    found_loop = true;
                                }
                                Transition::While(q_body, ns) => {
                                    if right_state == NIL_STATE_ID {
                                        // TODO: if q_for_action_trace.get_end() == action_trace.get_end() {
                                        // TODO: change the annotation of q0
                                        let q0_annotations =
                                            state_factory::get_state_by_id(curr_top_level_state)
                                                .annotations;
                                        assert!(q0_annotations.len() == 1);
                                        whiles_to_copy.push((
                                            curr_top_level_state,
                                            ns.clone(),
                                            *q_body,
                                            left_state,
                                        ));
                                    }
                                    found_loop = true;
                                }
                                Transition::ForData(vp, q_body) => {
                                    if right_state == NIL_STATE_ID {
                                        let q0_annotations =
                                            state_factory::get_state_by_id(curr_top_level_state)
                                                .annotations;
                                        assert!(q0_annotations.len() == 1);
                                        fordatas_to_copy.push((
                                            curr_top_level_state,
                                            vp.clone(),
                                            *q_body,
                                            left_state,
                                        ));
                                    }
                                    found_loop = true;
                                }
                                _ => {} // do nothing
                            }
                        }
                        if !found_loop {
                            if right_state == NIL_STATE_ID {
                                transitions_to_relink.push((curr_top_level_state, left_state));
                            } else {
                                next_top_level_state = right_state;
                            }
                        }
                    }
                    _ => {
                        panic!("Expecting a top-level state to have only Nil and Seq transitions underneath.")
                    }
                }
            }
            if old_top_level_state == next_top_level_state {
                break;
            } else {
                curr_top_level_state = next_top_level_state;
            }
        }

        self.increment_annotations();

        // append the new action
        let n = dom_trace.number_of_doms;
        let new_seq_annotations = vec![IOPair {
            input: None,
            output: ActionTrace(n - 1, n),
        }];
        let new_seq_state =
            create_new_state_with_annotations(GrammarSymbol::Program, new_seq_annotations.clone());
        // Set new seq as the final state at the start of synthesis
        if self.root == NIL_STATE_ID {
            self.root = new_seq_state;
        }
        // After the first action, we add the new seq to the fta by relinking
        else {
            for (target_state, left_state_id) in transitions_to_relink {
                self.transitions
                    .remove_transition(target_state, &Transition::Seq(left_state_id, NIL_STATE_ID));
                // We are gonna set the top-level transition later so no need to remove it here
                self.add_transition(Transition::Seq(left_state_id, new_seq_state), target_state);
                set_search_tree_top_level_transition(left_state_id, new_seq_state, target_state);
            }
        }

        // Link the new action state and Nil state to the new Seq
        let new_seq_left_state =
            create_new_state_with_annotations(GrammarSymbol::Expr, new_seq_annotations);
        self.add_transition(
            action.to_transition(matrix, config, num_selectors, 0),
            new_seq_left_state,
        );
        // Don't add expression transitions to search tree
        self.add_transition(
            Transition::Seq(new_seq_left_state, NIL_STATE_ID),
            new_seq_state,
        );
        set_search_tree_top_level_transition(new_seq_left_state, NIL_STATE_ID, new_seq_state);
        let mut for_ftas = vec![];
        // remove all transitions to the nil state
        for (parent_seq_state, q_body, prev_for_state) in fors_to_copy {
            let q_for = create_new_state(GrammarSymbol::Expr);
            // Transition::For(big_ns, q_body_), for_state)
            let (_, mut transitions_) = self.deep_copy(q_body, false);
            transitions_.insert_inplace(Transition::For(q_body), q_for);
            let for_fta = Fta {
                root: q_for,
                transitions: transitions_,
            };
            for_ftas.push((for_fta, q_body, q_for, parent_seq_state));

            self.transitions.remove_transition(
                parent_seq_state,
                &Transition::Seq(prev_for_state, NIL_STATE_ID),
            );
            remove_search_tree_loop_transition(prev_for_state, NIL_STATE_ID, parent_seq_state);
        }

        let mut while_ftas = vec![];
        // remove all transitions to the nil state
        for (parent_seq_state, ns, q_body, prev_for_state) in whiles_to_copy {
            let q_while = create_new_state(GrammarSymbol::Expr);
            let (_, mut transitions_) = self.deep_copy(q_body, false);
            transitions_.insert_inplace(Transition::While(q_body, ns.clone()), q_while);
            let while_fta = Fta {
                root: q_while,
                transitions: transitions_,
            };
            while_ftas.push((while_fta, ns, q_body, q_while, parent_seq_state));
            self.transitions.remove_transition(
                parent_seq_state,
                &Transition::Seq(prev_for_state, NIL_STATE_ID),
            );
            remove_search_tree_loop_transition(prev_for_state, NIL_STATE_ID, parent_seq_state);
        }

        let mut fordata_ftas = vec![];
        // remove all transitions to the nil state
        for (parent_seq_state, vp, q_body, prev_for_state) in fordatas_to_copy {
            let q_fordata = create_new_state(GrammarSymbol::Expr);
            let (_, mut transitions_) = self.deep_copy(q_body, false);
            transitions_.insert_inplace(Transition::ForData(vp.clone(), q_body), q_fordata);
            let fordata_fta = Fta {
                root: q_fordata,
                transitions: transitions_,
            };
            fordata_ftas.push((fordata_fta, vp, q_body, q_fordata, parent_seq_state));
            self.transitions.remove_transition(
                parent_seq_state,
                &Transition::Seq(prev_for_state, NIL_STATE_ID),
            );
            remove_search_tree_loop_transition(prev_for_state, NIL_STATE_ID, parent_seq_state);
        }

        // search for the q0' that has the same action_trace as the q0, the top level seq over the for state.
        // re-connect all seq transitions with left child being a for program and right child being nil
        let mut speculated_transitions = Transitions::empty();
        // println!("number_of_for: {}", for_ftas.len());
        let mut predictable = false;
        for (for_fta, q_body, q_for, q0) in for_ftas {
            if start_time.elapsed() >= time_left {
                return None;
            }
            let validate_for_results = validate_candidate_for(
                self, for_fta, q0, q_for, q_body, None, actions, matrices, false,
            );

            for result in validate_for_results {
                match result {
                    ValidateCandidateLoopResult::LinkedToLast(transitions) => {
                        predictable = true;
                        speculated_transitions.union_inplace(transitions);
                    }
                    ValidateCandidateLoopResult::LinkedToMiddle(transitions) => {
                        speculated_transitions.union_inplace(transitions);
                    }
                }
            }
        }
        for (while_fta, ns, q_body, q_while, q0) in while_ftas {
            if start_time.elapsed() >= time_left {
                return None;
            }
            // println!(
            //     "while:\n{}ms\n{}",
            //     start_time.elapsed().as_millis(),
            //     program_to_string(&while_fta.extract_program(q_body).0, "")
            // );
            let validate_while_results = validate_candidate_while(
                while_fta, q_while, q_body, q0, None, None, &ns, actions, matrices, false,
            );

            for result in validate_while_results {
                match result {
                    ValidateCandidateLoopResult::LinkedToLast(transitions) => {
                        speculated_transitions.union_inplace(transitions);
                        predictable = true;
                    }
                    ValidateCandidateLoopResult::LinkedToMiddle(transitions) => {
                        speculated_transitions.union_inplace(transitions);
                    }
                }
            }
        }
        for (fordata_fta, vp, q_body, q_fordata, q0) in fordata_ftas {
            if start_time.elapsed() >= time_left {
                return None;
            }
            let validate_fordata_results = validate_candidate_fordata(
                fordata_fta,
                q_fordata,
                q_body,
                q0,
                None,
                actions,
                matrices,
                data,
                &vp,
            );

            for result in validate_fordata_results {
                match result {
                    ValidateCandidateLoopResult::LinkedToLast(transitions) => {
                        speculated_transitions.union_inplace(transitions);
                        predictable = true;
                    }
                    ValidateCandidateLoopResult::LinkedToMiddle(transitions) => {
                        speculated_transitions.union_inplace(transitions);
                    }
                }
            }
        }
        self.transitions.union_inplace(speculated_transitions);
        Some(predictable)
    }
}
