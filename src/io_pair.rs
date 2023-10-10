use crate::env::env_to_string;

use super::{action::ActionTrace, dom::DomTrace, env::Env};

pub type IOPairInput = (DomTrace, Env);
pub type IOPairOutput = (ActionTrace, DomTrace);

/// An IO pair that records the input and output
/// when evaluating an Transition
#[derive(Hash, Clone, PartialOrd, Ord, PartialEq, Eq, Debug)]
pub struct IOPair {
    pub input: Env,
    pub output: ActionTrace,
}

impl std::fmt::Display for IOPair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} |-> {}", env_to_string(&self.input), self.output,)
    }
}
