use itertools::Itertools;
use smallvec::smallvec;

use crate::selector::{Index, Indices};

/// A context used to hole an environment of selector variables
/// to concrete selectors
///
/// Used during the evaluation of For.
/// Implemented as a Vec which acts as an efficient stack
pub type Env = Option<Indices>;

pub fn extend_context(context: &Env, i: Index) -> Env {
    Some(context.as_ref().map_or_else(
        || smallvec![i],
        |indices| {
            let mut indices = indices.clone();
            indices.push(i);
            indices
        },
    ))
}

pub fn env_to_string(env: &Env) -> String {
    env.as_ref()
        .map(|indices| indices.iter().map(Index::to_string).join(", "))
        .unwrap_or("[]".to_string())
}
