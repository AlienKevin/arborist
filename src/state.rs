use super::{annotation::*, grammar_symbol::*};
use std::hash::{Hash, Hasher};

/// Id of a `State`
///
/// Each state has a unique id within our `Fta`
pub type StateId = usize;

pub const NIL_STATE_ID: usize = 0;

/// A state in our Fta
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct State {
    pub grammar_symbol: GrammarSymbol,
    pub annotations: Annotations,
}

pub fn dummy_state_id() -> StateId {
    0
}
