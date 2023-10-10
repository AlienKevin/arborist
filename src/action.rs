use crate::{
    dom::DomTrace,
    dsl::{NullaryOp, UnaryOp},
    selector::{Index, Selector},
    selector_matrix::ConcreteSelectorMatrix,
    test_utils::BenchmarkConfig,
    transition::Transition,
    value_path::ValuePath,
};

/// Represents an action performed by the user on the browser frontend
#[derive(Hash, Clone, PartialEq, Eq, Debug)]
pub enum Action {
    Nullary(NullaryOp),
    Unary(Index, UnaryOp),
    SendKeys(Index, String),
    SendData(Index, ValuePath),
    // TODO: add the rest of actions from WebRobot paper
}

impl Action {
    pub fn to_transition(
        &self,
        matrix: &ConcreteSelectorMatrix,
        config: &BenchmarkConfig,
        num_selectors: Option<usize>,
        index: u64,
    ) -> Transition {
        match self {
            Self::Nullary(op) => Transition::Nullary(*op),
            Self::Unary(_matrix_index, op) => {
                let selectors = if let Some(num_selectors) = num_selectors {
                    matrix.get_num_selectors(num_selectors)
                } else {
                    matrix.get_selectors(config.max_selector_depth, config.sample_rate, index)
                };
                Transition::Unary(*op, selectors.into_iter().map(Selector::Concrete).collect())
            }
            Self::SendKeys(_matrix_index, str) => {
                let selectors = if let Some(num_selectors) = num_selectors {
                    matrix.get_num_selectors(num_selectors)
                } else {
                    matrix.get_selectors(config.max_selector_depth, config.sample_rate, index)
                };
                Transition::SendKeys(
                    selectors.into_iter().map(Selector::Concrete).collect(),
                    str.clone(),
                )
            }
            Self::SendData(_matrix_index, v) => {
                let selectors = if let Some(num_selectors) = num_selectors {
                    matrix.get_num_selectors(num_selectors)
                } else {
                    matrix.get_selectors(config.max_selector_depth, config.sample_rate, index)
                };
                Transition::SendData(
                    selectors.into_iter().map(Selector::Concrete).collect(),
                    v.clone(),
                )
            }
        }
    }
}

impl std::fmt::Display for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Action::Nullary(op) => {
                write!(f, "{}", op)
            }
            Action::Unary(n, op) => {
                write!(f, "{}({})", op, n)
            }
            Action::SendKeys(n, str) => {
                write!(f, "SendKeys({}, {})", n, str)
            }
            Action::SendData(n, v) => {
                write!(f, "SendData({}, {})", n, v)
            }
        }
    }
}

/// A list of actions performed by the user in sequence
pub type Actions = Vec<Action>;

/// Store all user actions with a `start_index` and `end_index` (exclusive)
/// that refers to the current actions during evaluation
#[derive(Hash, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct ActionTrace(pub Index, pub Index);

impl ActionTrace {
    /// Initialize an ActionTrace with a given length
    pub fn new(n: Index) -> Self {
        ActionTrace(0, n)
    }

    /// Create an empty ActionTrace
    pub fn empty() -> Self {
        ActionTrace(0, 0)
    }

    pub fn empty_at(start: Index) -> Self {
        ActionTrace(start, start)
    }

    /// Check if an ActionTrace is empty
    pub fn is_empty(&self) -> bool {
        self.0 == self.1
    }

    pub fn increment(&self) -> Self {
        ActionTrace(self.0, self.1 + 1)
    }

    /// Empty an ActionTrace but keep its end_index
    /// so that we can keep track of where
    /// we are in the overall actions list.
    /// Next time we call next(), we can
    /// move on to the next action based on
    /// the end_index saved.
    pub fn emptied(&self) -> Self {
        if self.is_empty() {
            *self
        } else {
            ActionTrace(self.1, self.1)
        }
    }

    // ActionTrace must end at the last action
    // In other words, the end action index = number_of_doms
    pub fn to_start_dom_trace(&self) -> DomTrace {
        DomTrace {
            n: self.0,
            number_of_doms: self.1,
        }
    }

    pub fn get_start(&self) -> Index {
        self.0
    }

    pub fn get_end(&self) -> Index {
        self.1
    }

    pub fn len(&self) -> Index {
        self.get_end() - self.get_start()
    }

    /// Check whether the next action in the action trace is consistent with
    /// the current action in our actions list, i.e. consistent(a, A, pi)
    pub fn is_consistent_with(&self, action: &Action, actions: &Actions) -> bool {
        // no more actions in expected action trace
        // inconsistent by definition
        if self.1 >= actions.len() as Index {
            // the action trace will never be updated to be longer than the actions
            // yet in prediction, we may have more actions than given action list
            false // TODO: verify this. changed back to false 2.15
        } else {
            let expected = &actions[self.1 as usize];
            // println!("expected: {}, actual: {}", expected, action);
            use Action::*;
            match (expected, action) {
                (Nullary(op1), Nullary(op2)) => op1 == op2,
                (Unary(i1, op1), Unary(i2, op2)) => i1 == i2 && op1 == op2,
                (SendKeys(i1, str1), SendKeys(i2, str2)) => i1 == i2 && str1 == str2,
                (SendData(i1, v1), SendData(i2, v2)) => i1 == i2 && v1 == v2,
                _ => false,
            }
        }
    }

    /// Get the next single action after this ActionTrace
    pub fn next(&self) -> ActionTrace {
        ActionTrace(self.1, self.1 + 1)
    }

    /// Merge two consecutive action traces
    pub fn merge(&self, other: &ActionTrace) -> ActionTrace {
        if self.is_empty() {
            *other
        } else if other.is_empty() {
            *self
        } else {
            if self.1 != other.0 {
                println!("self: {:?}", self);
                println!("other: {:?}", other);
            }
            assert!(self.1 == other.0);
            ActionTrace(self.0, other.1)
        }
    }
}

impl std::fmt::Display for ActionTrace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[a{}, a{})", self.0, self.1)
    }
}
