use super::{action::*, dom::*, env::Env, io_pair::*, state::StateId};
use crate::dsl::{NullaryOp, UnaryOp};
use crate::env::extend_context;
use crate::fta::Fta;
use crate::selector::Selector;
use crate::selector_matrix::{validate_selector, Matrices};
use crate::transition::Transition;
use crate::value_path::ValuePath;
use im_rc::Vector;

impl Fta {
    pub fn validate(
        &self,
        action_trace: ActionTrace,
        actions: &Actions,
        matrices: &Matrices,
        dom_trace: DomTrace,
        context: &Env,
        q: StateId,
    ) -> Option<IOPairOutput> {
        use Transition::*;
        if dom_trace.is_empty() {
            // Rule (2)
            Some((action_trace.emptied(), dom_trace))
        } else {
            let input_alphabets = self.transitions.get_by_id(q).unwrap_or_else(|| {
                panic!("Must have a transition to the input state with id {}.", q)
            });
            input_alphabets
                .iter()
                .find_map(|input_alphabet| match input_alphabet {
                    // Rule (3)
                    &Seq(q1, q2) => self.validate_seq(
                        action_trace,
                        actions,
                        matrices,
                        dom_trace,
                        context,
                        q1,
                        q2,
                    ),
                    &Nullary(op) => self.validate_nullary(action_trace, actions, dom_trace, op),
                    // Rule (4) and (5)
                    Unary(op, ns) => self.validate_unary(
                        action_trace,
                        actions,
                        matrices,
                        dom_trace,
                        context,
                        *op,
                        ns,
                    ),
                    SendKeys(ns, s) => self.validate_sendkeys(
                        action_trace,
                        actions,
                        matrices,
                        dom_trace,
                        context,
                        ns,
                        s,
                    ),
                    SendData(ns, vp) => self.validate_senddata(
                        action_trace,
                        actions,
                        matrices,
                        dom_trace,
                        context,
                        ns,
                        vp,
                    ),
                    For(q3) => {
                        self.validate_for(action_trace, actions, matrices, dom_trace, context, *q3)
                    }
                    While(q_body, ns) => {
                        self.validate_while(action_trace, actions, matrices, dom_trace, *q_body, ns)
                    }
                    Nil => {
                        // similar to Rule (2)
                        Some((action_trace.emptied(), dom_trace))
                    }
                    _ => panic!("Impossible input transition during validation"),
                })
        }
    }

    fn validate_seq(
        &self,
        action_trace: ActionTrace,
        actions: &Actions,
        matrices: &Matrices,
        dom_trace: DomTrace,
        context: &Env,
        q1: StateId,
        q2: StateId,
    ) -> Option<IOPairOutput> {
        // let k = self.transitions.get_by_id(q1).unwrap().iter();

        let results1 = self.validate(action_trace, actions, matrices, dom_trace, context, q1);
        let t1 = self
            .transitions
            .get_by_id(q1)
            .unwrap()
            .iter()
            .next()
            .unwrap();
        if matches!(
            t1,
            Transition::For(..) | Transition::While(..) | Transition::ForData(..)
        ) {
            return results1;
        }
        results1.and_then(|result1| {
            let (action_trace1, dom_trace1) = result1;
            let results2 = self.validate(action_trace1, actions, matrices, dom_trace1, context, q2);
            results2.map(|result2| {
                let (action_trace2, dom_trace2) = result2;
                (action_trace1.merge(&action_trace2), dom_trace2)
            })
        })
    }

    fn validate_nullary(
        &self,
        action_trace: ActionTrace,
        actions: &Actions,
        dom_trace: DomTrace,
        op: NullaryOp,
    ) -> Option<IOPairOutput> {
        assert!(!dom_trace.is_empty());

        if !action_trace.is_consistent_with(&Action::Nullary(op), actions) {
            None
        } else {
            let action_trace_ = action_trace.next();
            let dom_trace_ = dom_trace.next();
            Some((action_trace_, dom_trace_))
        }
    }

    fn validate_unary(
        &self,
        action_trace: ActionTrace,
        actions: &Actions,
        matrices: &Matrices,
        dom_trace: DomTrace,
        context: &Env,
        op: UnaryOp,
        ns: &Vector<Selector>,
    ) -> Option<IOPairOutput> {
        assert!(!dom_trace.is_empty());

        let matrix_index = dom_trace.n;
        if action_trace.is_consistent_with(&Action::Unary(matrix_index, op), actions) {
            let has_valid_n = ns.iter().any(|n| {
                let rho = n.to_concrete_selector(context);
                validate_selector(matrices, matrix_index, &rho)
            });

            if has_valid_n {
                let action_trace_ = action_trace.next();
                let dom_trace_ = dom_trace.next();
                Some((action_trace_, dom_trace_))
            } else {
                None
            }
        } else {
            None
        }
    }

    fn validate_sendkeys(
        &self,
        action_trace: ActionTrace,
        actions: &Actions,
        matrices: &Matrices,
        dom_trace: DomTrace,
        context: &Env,
        ns: &Vector<Selector>,
        s: &String,
    ) -> Option<IOPairOutput> {
        assert!(!dom_trace.is_empty());

        let matrix_index = dom_trace.n;
        if action_trace.is_consistent_with(&Action::SendKeys(matrix_index, s.clone()), actions) {
            let has_valid_n = ns.iter().any(|n| {
                let rho = n.to_concrete_selector(context);
                validate_selector(matrices, matrix_index, &rho)
            });

            if has_valid_n {
                let action_trace_ = action_trace.next();
                let dom_trace_ = dom_trace.next();
                Some((action_trace_, dom_trace_))
            } else {
                None
            }
        } else {
            None
        }
    }

    fn validate_senddata(
        &self,
        action_trace: ActionTrace,
        _actions: &Actions,
        _matrices: &Matrices,
        dom_trace: DomTrace,
        _context: &Env,
        _ns: &Vector<Selector>,
        _vp: &ValuePath,
    ) -> Option<IOPairOutput> {
        // TODO: Pass in data context to validate() so we can properly check for SendData
        assert!(!dom_trace.is_empty());
        let action_trace_ = action_trace.next();
        let dom_trace_ = dom_trace.next();
        Some((action_trace_, dom_trace_))
    }

    // TODO: Check that at least 1 abstract selector is valid
    pub fn validate_for(
        &self,
        action_trace: ActionTrace,
        actions: &Actions,
        matrices: &Matrices,
        dom_trace: DomTrace,
        context: &Env,
        q3: StateId,
    ) -> Option<IOPairOutput> {
        let body_context = extend_context(context, 1);
        self.validate(
            action_trace,
            actions,
            matrices,
            dom_trace,
            &body_context,
            q3,
        )
    }

    pub fn validate_while(
        &self,
        action_trace: ActionTrace,
        actions: &Actions,
        matrices: &Matrices,
        dom_trace: DomTrace,
        q_body: StateId,
        _ns: &Vector<Selector>,
    ) -> Option<IOPairOutput> {
        // TODO: also check if the click expr is valid (assuming a while must run for at least one iteration)
        let empty_context = None;
        self.validate(
            action_trace,
            actions,
            matrices,
            dom_trace,
            &empty_context,
            q_body,
        )
    }
}
