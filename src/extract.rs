use std::{cmp::Reverse, collections::BinaryHeap};

use rustc_hash::FxHashMap;

use crate::{
    dsl::{Expr, Program},
    fta::Fta,
    state::{StateId, NIL_STATE_ID},
    transition::Transition,
};

impl Fta {
    pub fn extract_program(&self, state_id: StateId) -> (Program, usize) {
        // Detect empty program at the start of incremental synthesis
        if self.transitions.get_by_id(state_id).is_none() {
            return (vec![], 0);
        }

        let mut pq = BinaryHeap::new();
        let mut visited: FxHashMap<usize, usize> = FxHashMap::default();
        let mut edge: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
        let mut found = false;
        pq.push((Reverse(0), state_id));
        visited.insert(state_id, 0);
        edge.insert(state_id, Vec::new());

        while let Some((Reverse(curr_length), curr_state)) = pq.pop() {
            if found && curr_length > visited[&NIL_STATE_ID] {
                break;
            }
            let transitions = self.transitions.get_by_id(curr_state).unwrap();
            for seq in transitions {
                match *seq {
                    Transition::Seq(_left, right) => {
                        let new_lengh = curr_length + 1;
                        if !visited.contains_key(&right) || new_lengh < visited[&right] {
                            pq.push((Reverse(new_lengh), right));
                            edge.entry(right).or_insert(vec![]).push(curr_state);
                            visited.insert(right, new_lengh);
                        }
                    }
                    Transition::Nil => {
                        visited.insert(NIL_STATE_ID, curr_length);
                        found = true;
                    }
                    _ => panic!("Expecting top-level Seqs. but got {:?}", seq),
                }
            }
        }
        if !found {
            return (vec![], 0);
        }

        // assemble the shortest program
        let mut path = vec![];
        let mut program_size = 0;

        let mut end_node = NIL_STATE_ID;
        while end_node != state_id {
            let mut shortest_expr: Option<Expr> = None;
            let mut shortest_expr_size: usize = usize::MAX;
            for pred in edge.get(&end_node).unwrap() {
                let transitions = self.transitions.get_by_id(*pred).unwrap();
                for seq in transitions {
                    match seq {
                        &Transition::Seq(left, right) => {
                            if right == end_node {
                                let (expr, expr_size) = self.get_shortest_expr(left);
                                if expr_size < shortest_expr_size {
                                    end_node = *pred;
                                    shortest_expr = Some(expr);
                                    shortest_expr_size = expr_size;
                                }
                                break;
                            }
                        }
                        _ => panic!("Only Seqs are expected here."),
                    }
                }
            }
            path.push(shortest_expr.unwrap());
            program_size += shortest_expr_size;
        }
        path.reverse();
        (path, program_size)
    }

    fn get_shortest_expr(&self, state_id: StateId) -> (Expr, usize) {
        let transitions = self.transitions.get_by_id(state_id).unwrap();
        let mut min_length = usize::MAX;
        let mut ret_expr = None;
        for transition in transitions {
            match transition {
                Transition::Unary(op, selectors) => {
                    return (Expr::Unary(selectors.clone(), *op), 1);
                }
                Transition::Nullary(op) => {
                    return (Expr::Nullary(*op), 1);
                }
                Transition::For(q_body) => {
                    let (body, prog_len) = self.extract_program(*q_body);
                    if prog_len < min_length {
                        min_length = prog_len;
                        ret_expr = Some(Expr::For(Box::new(body)));
                    }
                }
                Transition::While(q_body, ns) => {
                    assert!(!ns.is_empty()); // there should be at least one n
                    let (body, prog_len) = self.extract_program(*q_body);
                    if prog_len < min_length {
                        min_length = prog_len;
                        ret_expr = Some(Expr::While(Box::new(body), ns.clone()));
                    }
                }
                Transition::ForData(vp, q_body) => {
                    let (body, prog_len) = self.extract_program(*q_body);
                    if prog_len < min_length {
                        min_length = prog_len;
                        ret_expr = Some(Expr::ForData(vp.clone(), Box::new(body)));
                    }
                }
                Transition::SendKeys(selectors, data) => {
                    return (Expr::SendKeys(selectors.clone(), data.clone()), 1);
                }
                Transition::SendData(selectors, vp) => {
                    return (Expr::SendData(selectors.clone(), vp.clone()), 1);
                }
                _ => panic!("Unexpected transition {:?}", transition),
            }
        }
        (ret_expr.unwrap(), min_length)
    }
}
