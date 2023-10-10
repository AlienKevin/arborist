use itertools::Itertools;
use serde_json::Value;
use smallvec::{smallvec, SmallVec};

use crate::{
    action::{Action, Actions},
    dom_query,
    dsl::{Expr, NullaryOp, Program, UnaryOp},
    env::{extend_context, Env},
    selector::Index,
    selector::{ConcreteSelector, Indices, Selector, SelectorPath, SelectorPathRaw},
    selector_matrix::validate_selector,
    selector_matrix::Matrices,
    value_path::ValuePath,
    value_path::{self, get_list_at_path},
};

#[derive(Hash, Clone, PartialEq, Eq, Debug)]
pub enum Predict {
    Nullary(NullaryOp),
    Unary(Vec<ConcreteSelector>, UnaryOp),
    SendKeys(Vec<ConcreteSelector>, String),
    SendData(Vec<ConcreteSelector>, ValuePath),
    // TODO: add the rest of actions from WebRobot paper
}

impl std::fmt::Display for Predict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Predict::Nullary(op) => write!(f, "{}", op),
            Predict::Unary(_selector, op) => write!(f, "{}", op),
            Predict::SendKeys(_selector, keys) => write!(f, "SendKeys(\"{}\")", keys),
            Predict::SendData(_selector, path) => write!(f, "SendData({})", path),
        }
    }
}

pub fn run_program(
    program: &Program,
    start_step: Index,
    step: Index,
    context: &Env,
    data_context: &Option<ValuePath>,
    matrices: &Matrices,
    data: &Option<Value>,
    actions: &Actions,
) -> Option<Predict> {
    let (prediction, moved_steps) = run_program_helper(
        program,
        start_step,
        step,
        context,
        data_context,
        matrices,
        data,
        actions,
    );
    if moved_steps == step {
        prediction
    } else {
        None
    }
}

pub fn run_program_helper(
    program: &Program,
    start_step: Index,
    step: Index,
    context: &Env,
    data_context: &Option<ValuePath>,
    matrices: &Matrices,
    data: &Option<Value>,
    actions: &Actions,
) -> (Option<Predict>, Index) {
    let mut predict: Option<Predict> = None;
    let mut moved_steps = start_step;

    for expr in program {
        if moved_steps == step {
            return (predict, moved_steps);
        }
        match expr {
            Expr::Nullary(op) => {
                predict = Some(Predict::Nullary(*op));
                moved_steps += 1;
            }

            Expr::Unary(ns, op) => {
                let rhos = ns
                    .iter()
                    .map(|n| n.to_concrete_selector(context))
                    .collect_vec();
                predict = Some(Predict::Unary(rhos, *op));
                moved_steps += 1;
            }
            Expr::For(p) => {
                let mut index = 1;
                let mut body_context = extend_context(context, index);
                while validate_for(p, moved_steps, step, &body_context, matrices, actions) {
                    // println!("enter loop iteration index: {}", index);
                    let (final_predict, steps) = run_program_helper(
                        p,
                        moved_steps,
                        step,
                        &body_context,
                        data_context,
                        matrices,
                        data,
                        actions,
                    );
                    index += 1;
                    body_context = extend_context(context, index);
                    predict = final_predict;
                    moved_steps = steps;
                    if moved_steps == step {
                        break;
                    }
                }
            }
            Expr::While(p, ns) => loop {
                let (final_predict, steps) = run_program_helper(
                    p,
                    moved_steps,
                    step,
                    &None,
                    data_context,
                    matrices,
                    data,
                    actions,
                );
                predict = final_predict;
                moved_steps = steps;
                if moved_steps == step {
                    break;
                }
                let mut rhos = ns
                    .iter()
                    .map(|n| n.to_concrete_selector(context))
                    .collect_vec();
                rhos.retain(|rho| validate_selector(matrices, moved_steps, rho));
                if rhos.is_empty() {
                    break;
                }
                moved_steps += 1;
                if moved_steps == step {
                    predict = Some(Predict::Unary(rhos, UnaryOp::Click));
                    break;
                }
            },
            Expr::ForData(v, p) => {
                let data_ = data.as_ref().unwrap();
                let data_context_ = data_context.as_ref().unwrap();
                let body_context = data_context_.append(v);
                let data_len = if let Some(value_list) = get_list_at_path(data_, &body_context.0) {
                    value_list.len()
                } else {
                    0
                };
                let mut index = 0;
                while index < data_len {
                    let data_context = body_context.append_index(value_path::Index::Number(index));
                    // println!("enter loop iteration index: {}", index);
                    let (final_predict, steps) = run_program_helper(
                        p,
                        moved_steps,
                        step,
                        &None,
                        &Some(data_context),
                        matrices,
                        data,
                        actions,
                    );
                    index += 1;
                    predict = final_predict;
                    moved_steps = steps;
                    if moved_steps == step {
                        break;
                    }
                }
            }
            Expr::SendKeys(ns, data) => {
                let rhos = ns
                    .iter()
                    .map(|n| n.to_concrete_selector(context))
                    .collect_vec();
                predict = Some(Predict::SendKeys(rhos, data.clone()));
                moved_steps += 1;
            }
            Expr::SendData(ns, v) => {
                let rhos = ns
                    .iter()
                    .map(|n| n.to_concrete_selector(context))
                    .collect_vec();
                let data_context = data_context.as_ref().unwrap();
                let suffix = if data_context.len() >= v.0.len() {
                    ValuePath::new()
                } else {
                    ValuePath::from_path(v.0[data_context.len()..].to_vec())
                };
                let concrete_vp = data_context.clone().append(&suffix);

                predict = Some(Predict::SendData(rhos, concrete_vp));
                moved_steps += 1;
            }
        }
    }
    (predict, moved_steps)
}

pub fn validate_unary(
    s: &Selector,
    op: &UnaryOp,
    moved_steps: Index,
    context: &Env,
    matrices: &Matrices,
    actions: &Actions,
) -> bool {
    let expected = &actions[moved_steps as usize];
    match expected {
        Action::Unary(_, op2) => {
            if op != op2 {
                return false;
            }
            let rho = s.to_concrete_selector(context);
            validate_selector(matrices, moved_steps, &rho)
        }
        _ => false,
    }
}

fn get_abstract_selectors_from_program(program: &Program) -> Vec<(SelectorPath, Indices)> {
    get_abstract_selectors_from_program_helper(program).unwrap_or_default()
}

fn get_abstract_selectors_from_program_helper(
    program: &Program,
) -> Option<Vec<(SelectorPath, Indices)>> {
    use crate::predict::Expr::*;
    program.iter().find_map(|p| match p {
        Unary(ns, _) | SendKeys(ns, _) | SendData(ns, _) => {
            let abstracts = ns
                .iter()
                .filter_map(|n| match n {
                    Selector::Abstract(path, breakpoints) => {
                        // Extract big N
                        // Relying on big N to judge whether to enter a loop is problematic
                        // because chances are the big N will very likely remain on the page
                        // even when the loop is supposed to stop.
                        // Instead, we can use the first group of abstract selectors to judge
                        // whether to enter or not.

                        // let last_breakpoint = (*breakpoints.last().unwrap()) as usize + 1;
                        // let mut indices = indices.clone();
                        // indices.truncate(last_breakpoint);
                        // Some(SelectorPathRaw {
                        //     path: path.iter().take(last_breakpoint).cloned().collect(),
                        //     indices,
                        // })

                        Some((path.clone(), breakpoints.clone()))
                    }
                    Selector::Concrete(_) => None,
                })
                .collect_vec();
            if abstracts.is_empty() {
                None
            } else {
                Some(abstracts)
            }
        }
        For(body) => get_abstract_selectors_from_program_helper(body),
        Nullary(_) | While(..) | ForData(..) => None,
    })
}

pub fn validate_for(
    program: &Program,
    start_step: Index,
    step: Index,
    context: &Env,
    matrices: &Matrices,
    actions: &Actions,
) -> bool {
    if start_step == step - 1 {
        if let Some(indices) = context {
            println!("Prediction reached end of DOM");
            let abstracts = get_abstract_selectors_from_program(program);
            let queries = abstracts
                .into_iter()
                .map(|(path, breakpoints)| {
                    let concrete_n = if indices.len() >= breakpoints.len() {
                        let n = Selector::Abstract(path, breakpoints);
                        n.to_concrete_selector(context)
                    } else {
                        let mut indices = indices.clone();
                        indices.extend([1].repeat(breakpoints.len() - indices.len()));
                        let n = Selector::Abstract(path, breakpoints);
                        n.to_concrete_selector(&Some(indices))
                    };
                    let n = concrete_n.to_query_xpath();
                    // println!("Querying {n}");
                    n
                })
                .collect_vec();
            return dom_query::query(step, queries);
        }
    }

    let mut moved_steps = start_step;

    for expr in program {
        if moved_steps == step {
            return true;
        }
        match expr {
            Expr::Nullary(op) => {
                let expected = &actions[moved_steps as usize];
                match expected {
                    Action::Nullary(op2) => {
                        if op != op2 {
                            return false;
                        }
                    }
                    _ => {
                        return false;
                    }
                }
                moved_steps += 1;
            }

            Expr::Unary(ns, op) => {
                if !ns
                    .iter()
                    .any(|n| validate_unary(n, op, moved_steps, context, matrices, actions))
                {
                    return false;
                }
                moved_steps += 1;
            }
            Expr::SendKeys(ns, data) => {
                let expected = &actions[moved_steps as usize];
                match expected {
                    Action::SendKeys(_, data2) => {
                        if data != data2 {
                            return false;
                        }
                        if !ns.iter().any(|n| {
                            let rho = n.to_concrete_selector(context);
                            validate_selector(matrices, moved_steps, &rho)
                        }) {
                            return false;
                        }
                    }
                    _ => {
                        return false;
                    }
                }
                moved_steps += 1;
            }
            Expr::SendData(ns, v) => {
                let expected = &actions[moved_steps as usize];
                match expected {
                    Action::SendData(_, v2) => {
                        if v != v2 {
                            return false;
                        }
                        if !ns.iter().any(|n| {
                            let rho = n.to_concrete_selector(context);
                            validate_selector(matrices, moved_steps, &rho)
                        }) {
                            return false;
                        }
                    }
                    _ => {
                        return false;
                    }
                }
                moved_steps += 1;
            }

            Expr::For(p) => {
                let body_context = extend_context(context, 1);
                let valid = validate_for(p, moved_steps, step, &body_context, matrices, actions);
                return valid;
            }
            Expr::While(p, _ns) => {
                // TODO: also check if the click expr is valid (assuming a while must run for at least one iteration)
                return validate_for(p, moved_steps, step, &None, matrices, actions);
            }
            _ => {
                panic!("not implemented")
            }
        }
    }
    true
}

/// Check whether the prediction is consistent with the next action
pub fn check_prediction(action: &Action, predict: &Predict, matrices: &Matrices) -> bool {
    match (action, predict) {
        (Action::Nullary(op1), Predict::Nullary(op2)) => op1 == op2,
        (Action::Unary(i1, op1), Predict::Unary(ns, op2)) => {
            ns.iter().any(|n| validate_selector(matrices, *i1, n)) && op1 == op2
        }
        (Action::SendKeys(i1, str1), Predict::SendKeys(ns, str2)) => {
            ns.iter().any(|n| validate_selector(matrices, *i1, n)) && str1 == str2
        }
        (Action::SendData(i1, v1), Predict::SendData(ns, v2)) => {
            ns.iter().any(|n| validate_selector(matrices, *i1, n)) && v1 == v2
        }
        _ => false,
    }
}
