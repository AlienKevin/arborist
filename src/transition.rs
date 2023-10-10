use std::{
    fmt::{Debug, Display},
    hash::Hasher,
};

use im_rc::{hashmap::Entry, HashMap, HashSet, Vector};
use rustc_hash::FxHasher;
use std::hash::BuildHasherDefault;

use crate::{
    dsl::{NullaryOp, UnaryOp},
    selector::Selector,
    state::StateId,
    value_path::ValuePath,
};

type FxHashMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;
type FxHashSet<K> = HashSet<K, BuildHasherDefault<FxHasher>>;

/// A map of transitions with `StateId` as the key.
/// The values are all the `Transition`s that transitions
/// into a given `StateId` key
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Transitions(pub FxHashMap<StateId, FxHashSet<Transition>>);

impl Transitions {
    /// Create an empty `Transitions`
    pub fn empty() -> Self {
        Transitions(FxHashMap::default())
    }

    /// Get all `Transition`s transitioning into a given `StateId`
    pub fn get_by_id(&self, state_id: StateId) -> Option<&FxHashSet<Transition>> {
        self.0.get(&state_id)
    }

    /// Union two `Transitions`
    pub fn union(mut self, other: Transitions) -> Self {
        self.union_inplace(other);
        self
    }

    pub fn union_inplace(&mut self, other: Transitions) {
        if self.0.is_empty() {
            *self = other;
        } else {
            for (q, other_transitions) in other.0.into_iter() {
                match self.0.entry(q) {
                    Entry::Occupied(mut self_transitions) => {
                        if !self_transitions.get().ptr_eq(&other_transitions) {
                            self_transitions.get_mut().extend(other_transitions);
                        }
                    }
                    Entry::Vacant(self_transitions) => {
                        self_transitions.insert(other_transitions);
                    }
                }
            }
        }
    }

    /// Insert a new transition into `Transitions`
    pub fn insert(mut self, input_transition: Transition, output_state: StateId) -> Self {
        self.insert_inplace(input_transition, output_state);
        self
    }

    pub fn insert_inplace(&mut self, input_transition: Transition, output_state: StateId) {
        match self.0.entry(output_state) {
            Entry::Occupied(mut transitions) => {
                transitions.get_mut().insert(input_transition);
            }
            Entry::Vacant(transitions) => {
                transitions.insert(FxHashSet::from_iter([input_transition]));
            }
        }
    }

    pub fn remove_transition(&mut self, target_state: StateId, transition: &Transition) {
        self.0.entry(target_state).and_modify(|transitions| {
            transitions.remove(transition);
        });
    }

    pub fn size(&self) -> usize {
        self.0.len()
    }
}

/// The Transition of our FTA
///
/// Correspond roughly to `Expr` in our DSL plus selector and index.
/// `Nil` is used at the end of a `Seq`
/// TODO: add type guarantees to the states embedded in Click and For, etc
#[derive(PartialEq, Ord, PartialOrd, Eq, Clone)]
pub enum Transition {
    Seq(StateId, StateId),
    Nil,
    Nullary(NullaryOp),
    Unary(UnaryOp, Vector<Selector>),
    For(StateId),
    While(StateId, Vector<Selector>),
    SendKeys(Vector<Selector>, String),
    SendData(Vector<Selector>, ValuePath),
    ForData(ValuePath, StateId),
}

impl std::hash::Hash for Transition {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Transition::Seq(state_id1, state_id2) => {
                state.write_u64(0); // Identifier for the Seq variant
                state_id1.hash(state);
                state_id2.hash(state);
            }
            Transition::Nil => {
                state.write_u64(1); // Identifier for the Nil variant
            }
            Transition::Nullary(op) => {
                state.write_u64(2); // Identifier for the Nullary variant
                op.hash(state);
            }
            Transition::Unary(op, _selectors) => {
                state.write_u64(3); // Identifier for the Unary variant
                op.hash(state);
                // TODO: confirm we can skip hashing selectors
            }
            Transition::For(state_id) => {
                state.write_u64(4); // Identifier for the For variant
                state_id.hash(state);
            }
            Transition::While(state_id, _selectors) => {
                state.write_u64(5); // Identifier for the While variant
                state_id.hash(state);
                // TODO: confirm we can skip hashing selectors
            }
            Transition::SendKeys(_selectors, string) => {
                state.write_u64(6); // Identifier for the SendKeys variant
                string.hash(state);
                // TODO: confirm we can skip hashing selectors
            }
            Transition::SendData(_selectors, value_path) => {
                state.write_u64(7); // Identifier for the SendData variant
                value_path.hash(state);
                // TODO: confirm we can skip hashing selectors
            }
            Transition::ForData(value_path, state_id) => {
                state.write_u64(8); // Identifier for the ForData variant
                value_path.hash(state);
                state_id.hash(state);
            }
        }
    }
}

impl Display for Transition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Seq(q1, q2) => write!(f, "Seq({}, {})", q1, q2),
            Self::Nil => write!(f, "Nil"),
            Self::Nullary(_) => write!(f, "Nullary"),
            Self::Unary(op, ns) => write!(f, "Unary({}, {})", op, ns.len()),
            Self::For(q_body) => write!(f, "For({})", q_body),
            Self::While(q_body, ns) => write!(f, "While({} ns, {})", ns.len(), q_body),
            Self::SendKeys(ns, s) => write!(f, "SendKeys({} ns, {})", ns.len(), s),
            Self::SendData(ns, vp) => write!(f, "SendData({} ns, {})", ns.len(), vp),
            Self::ForData(vp, q_body) => write!(f, "ForData({}, {})", vp, q_body),
        }
    }
}

impl Debug for Transition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string())
    }
}
