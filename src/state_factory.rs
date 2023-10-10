use crate::action::ActionTrace;
use crate::annotation::Annotations;
use crate::dom::DomTrace;
use crate::fta::Fta;
use crate::grammar_symbol::GrammarSymbol;
use crate::io_pair::IOPair;
use crate::selector::Index;
use crate::state::{State, StateId, NIL_STATE_ID};
use crate::synthesis::ActionType;
use itertools::Itertools;
use rustc_hash::{FxHashMap, FxHashSet, FxHasher};
use std::cell::RefCell;
use std::hash::{Hash, Hasher};
use std::mem;

#[derive(Clone, Debug, PartialEq, Hash, Eq)]
pub struct SearchTreeEntry {
    pub left: StateId,
    pub target: StateId,
}

#[derive(Default, Clone)]
pub struct StateFactory {
    states: FxHashMap<StateId, State>,
    state_factory: FxHashMap<u64, StateId>,
    pub max_state_id: StateId,
    copied_states: FxHashMap<StateId, StateId>,
    fta_search_tree: FxHashMap<StateId, (SearchTreeEntry, FxHashSet<SearchTreeEntry>)>, // mapped to top-level Seq state + loop states
    fta_search_cache: FxHashMap<(usize, StateId), Vec<(Vec<(StateId, StateId)>, StateId)>>,
    is_same_action_type_cache: FxHashMap<(StateId, StateId), (bool, bool)>,
    first_while_pattern: FxHashMap<StateId, Vec<ActionType>>,
    while_body_last_stopped_dom: FxHashMap<StateId, Index>,
    for_body_last_stopped_dom: FxHashMap<StateId, Index>,
    is_early_stopped: bool,
}

impl StateFactory {
    pub fn new() -> Self {
        StateFactory {
            states: FxHashMap::default(),
            state_factory: FxHashMap::default(),
            max_state_id: 0,
            copied_states: FxHashMap::default(),
            fta_search_tree: FxHashMap::default(),
            fta_search_cache: FxHashMap::default(),
            is_same_action_type_cache: FxHashMap::default(),
            first_while_pattern: FxHashMap::default(),
            while_body_last_stopped_dom: FxHashMap::default(),
            for_body_last_stopped_dom: FxHashMap::default(),
            is_early_stopped: false,
        }
    }

    pub fn set_first_while_pattern(&mut self, q0: StateId, pattern: Vec<ActionType>) {
        self.first_while_pattern.entry(q0).or_insert_with(|| {
            println!(
                "Set first while pattern to {:?} starting at state {}",
                pattern, q0
            );
            pattern
        });
    }

    pub fn get_first_while_pattern(&self, q0: StateId) -> Option<Vec<ActionType>> {
        self.first_while_pattern.get(&q0).cloned()
    }

    pub fn get_all_first_while_patterns(&self) -> Vec<Vec<ActionType>> {
        self.first_while_pattern.values().cloned().collect_vec()
    }

    pub fn remove_state_by_id(&mut self, state_id: StateId) {
        let state = self.states.get(&state_id).unwrap();
        let key_hash = Self::hash_factory_key(&state.grammar_symbol, &state.annotations);
        self.state_factory.remove(&key_hash);
        self.states.remove(&state_id);
    }

    pub fn clean_up_states(&mut self, reachable_states: FxHashSet<StateId>) {
        let mut states = mem::take(&mut self.states);
        self.states = reachable_states
            .iter()
            .map(|reachable_state| states.remove_entry(reachable_state).unwrap())
            .collect();
        let state_factory = mem::take(&mut self.state_factory);
        self.state_factory = state_factory
            .into_iter()
            .filter(|(_, state_id)| reachable_states.contains(state_id))
            .collect();
        self.copied_states.clear();
    }

    fn hash_factory_key(grammar_symbol: &GrammarSymbol, annotations: &Annotations) -> u64 {
        let mut s = FxHasher::default();
        grammar_symbol.hash(&mut s);
        annotations.hash(&mut s);
        s.finish()
    }

    pub fn new_state_with_annotations(&mut self, key: (GrammarSymbol, Annotations)) -> StateId {
        let key_hash = Self::hash_factory_key(&key.0, &key.1);
        match self.state_factory.get(&key_hash) {
            Some(state) => *state,
            None => {
                let grammar_symbol = key.0;
                let annotations = key.1;
                let new_state = State {
                    grammar_symbol,
                    annotations,
                };
                let new_state_id = self.max_state_id;
                self.states.insert(new_state_id, new_state);
                self.state_factory.insert(key_hash, new_state_id);
                self.max_state_id += 1;
                new_state_id
            }
        }
    }

    pub fn new_state(&mut self, grammar_symbol: GrammarSymbol) -> StateId {
        let new_state = State {
            grammar_symbol,
            annotations: vec![],
        };
        let new_state_id = self.max_state_id;
        self.states.insert(self.max_state_id, new_state);
        self.max_state_id += 1;
        new_state_id
    }

    /// Get an annotated state if it is already present in our `state_factory`.
    /// Otherwise, create a new state and add it to our `state_factory`.
    pub fn get_annotated_new_state(
        &mut self,
        grammar_symbol: GrammarSymbol,
        state: StateId,
        annotation: IOPair,
    ) -> StateId {
        let old_annotations = &(self.get_state_by_id(state).annotations);
        if !old_annotations.is_empty()
            && old_annotations.last().unwrap().output.0 == annotation.output.0
        {
            if old_annotations.last().unwrap().output.1 == annotation.output.1 {
                state
            } else {
                let mut new_annotations = old_annotations.clone();
                new_annotations[old_annotations.len() - 1] = annotation;
                self.new_state_with_annotations((grammar_symbol, new_annotations))
            }
        } else {
            let mut new_annotations = old_annotations.clone();
            new_annotations.push(annotation);
            self.new_state_with_annotations((grammar_symbol, new_annotations))
        }
    }

    pub fn copy_state(&mut self, state_id: StateId) -> StateId {
        if state_id == NIL_STATE_ID {
            state_id
        } else {
            match self.copied_states.get(&state_id) {
                Some(copied_state_id) => *copied_state_id,
                None => {
                    let grammar_symbol = self.get_state_by_id(state_id).grammar_symbol;
                    let copied_state_id = self.new_state(grammar_symbol);
                    self.copied_states.insert(state_id, copied_state_id);
                    copied_state_id
                }
            }
        }
    }

    pub fn get_state_by_id(&self, state_id: StateId) -> &State {
        self.states.get(&state_id).unwrap()
    }

    pub fn get_last_output(
        &self,
        fta: &Fta,
        number_of_doms: Index,
    ) -> Option<(ActionTrace, DomTrace)> {
        if fta.transitions.0.is_empty() || fta.root == NIL_STATE_ID {
            None
        } else {
            let action_trace = self
                .get_state_by_id(fta.root)
                .annotations
                .last()
                .unwrap()
                .output;
            Some((
                action_trace,
                DomTrace {
                    number_of_doms,
                    n: action_trace.1,
                },
            ))
        }
    }

    pub fn clear(&mut self) {
        self.states.clear();
        self.state_factory.clear();
        self.copied_states.clear();
        self.fta_search_tree.clear();
        self.fta_search_cache.clear();
        self.is_same_action_type_cache.clear();
        self.first_while_pattern.clear();
        self.while_body_last_stopped_dom.clear();
        self.for_body_last_stopped_dom.clear();
        self.is_early_stopped = false;
        self.max_state_id = 0;
    }

    pub fn clear_annotations(&mut self) {
        for (_, state) in self.states.iter_mut() {
            state.annotations.clear();
        }
        self.state_factory.clear();
    }

    pub fn increment_annotation(&mut self, state_id: StateId) {
        // Remove old annotations mapped to the state
        let state = self.states.get(&state_id).unwrap();
        self.state_factory.remove(&Self::hash_factory_key(
            &state.grammar_symbol,
            &state.annotations,
        ));

        self.states.entry(state_id).and_modify(|state| {
            assert_eq!(state.annotations.len(), 1);
            let output_action_trace = state.annotations[0].output;
            state.annotations[0].output = output_action_trace.increment();
            // Map new annotations to the state
            self.state_factory.insert(
                Self::hash_factory_key(&state.grammar_symbol, &state.annotations),
                state_id,
            );
        });
    }

    pub fn add_search_tree_loop_transition(
        &mut self,
        left: StateId,
        right: StateId,
        target: StateId,
    ) {
        self.fta_search_tree
            .entry(right)
            .and_modify(|(_top_level, loop_transitions)| {
                loop_transitions.insert(SearchTreeEntry { left, target });
            })
            .or_insert_with(|| {
                panic!("Must insert top-level Seq state first before inserting loop states.")
            });
    }

    pub fn set_search_tree_top_level_transition(
        &mut self,
        left: StateId,
        right: StateId,
        target: StateId,
    ) {
        self.fta_search_tree
            .entry(right)
            .and_modify(|(top_level, _loop_transitions)| {
                *top_level = SearchTreeEntry { left, target };
            })
            .or_insert((SearchTreeEntry { left, target }, FxHashSet::default()));
    }

    pub fn remove_search_tree_loop_transition(
        &mut self,
        left: StateId,
        right: StateId,
        target: StateId,
    ) {
        self.fta_search_tree
            .entry(right)
            .and_modify(|(_top_level, loop_transitions)| {
                loop_transitions.remove(&SearchTreeEntry { left, target });
            });
    }

    pub fn get_search_tree_transition(
        &self,
        right: StateId,
    ) -> Option<(SearchTreeEntry, FxHashSet<SearchTreeEntry>)> {
        self.fta_search_tree.get(&right).cloned()
    }

    pub fn set_is_same_action_type(
        &mut self,
        p1: StateId,
        p2: StateId,
        is_same_action_type: bool,
        has_senddata: bool,
    ) {
        self.is_same_action_type_cache
            .insert((p1, p2), (is_same_action_type, has_senddata));
    }

    pub fn get_is_same_action_type(&self, p1: StateId, p2: StateId) -> Option<(bool, bool)> {
        self.is_same_action_type_cache.get(&(p1, p2)).copied()
    }

    pub fn set_while_body_last_stopped_dom(&mut self, q_body: StateId, n: Index) {
        self.while_body_last_stopped_dom.insert(q_body, n);
    }

    pub fn get_while_body_last_stopped_dom(&self, q_body: StateId) -> Option<Index> {
        self.while_body_last_stopped_dom.get(&q_body).copied()
    }

    pub fn set_for_body_last_stopped_dom(&mut self, q_body: StateId, n: Index) {
        self.for_body_last_stopped_dom.insert(q_body, n);
    }

    pub fn get_for_body_last_stopped_dom(&self, q_body: StateId) -> Option<Index> {
        self.for_body_last_stopped_dom.get(&q_body).copied()
    }

    pub fn set_is_early_stopped(&mut self, stopped: bool) {
        self.is_early_stopped = stopped;
    }

    pub fn get_is_early_stopped(&self) -> bool {
        self.is_early_stopped
    }

    pub fn get_states_size(&self) -> usize {
        self.states.len()
    }

    pub fn get_state_factory_size(&self) -> usize {
        self.state_factory.len()
    }
}

/// Create a new state without any annotation
///
/// Used in testing
pub fn create_new_state(grammar_symbol: GrammarSymbol) -> StateId {
    STATE_FACTORY.with(|f| f.borrow_mut().new_state(grammar_symbol))
}

pub fn create_new_state_with_annotations(
    grammar_symbol: GrammarSymbol,
    annotations: Annotations,
) -> StateId {
    STATE_FACTORY.with(|f| {
        f.borrow_mut()
            .new_state_with_annotations((grammar_symbol, annotations))
    })
}

pub fn get_annotated_new_state(
    grammar_symbol: GrammarSymbol,
    state: StateId,
    annotation: IOPair,
) -> usize {
    STATE_FACTORY.with(|f| {
        f.borrow_mut()
            .get_annotated_new_state(grammar_symbol, state, annotation)
    })
}

pub fn copy_state(state_id: StateId) -> StateId {
    STATE_FACTORY.with(|f| f.borrow_mut().copy_state(state_id))
}

pub fn get_state_by_id(state_id: StateId) -> State {
    STATE_FACTORY.with(|f| f.borrow().get_state_by_id(state_id).clone())
}

pub fn get_last_output(fta: &Fta, number_of_doms: Index) -> Option<(ActionTrace, DomTrace)> {
    STATE_FACTORY.with(|f| f.borrow().get_last_output(fta, number_of_doms))
}

pub fn clear_state_factory() {
    STATE_FACTORY.with(|f| f.borrow_mut().clear());
}

pub fn clear_annotations() {
    STATE_FACTORY.with(|f| f.borrow_mut().clear_annotations());
}

pub fn increment_annotation(state_id: StateId) {
    STATE_FACTORY.with(|f| f.borrow_mut().increment_annotation(state_id));
}

pub fn clean_up_states(reachable_states: FxHashSet<StateId>) {
    STATE_FACTORY.with(|f| f.borrow_mut().clean_up_states(reachable_states));
}

pub fn get_states_size() -> usize {
    STATE_FACTORY.with(|f| f.borrow().get_states_size())
}

pub fn get_state_factory_size() -> usize {
    STATE_FACTORY.with(|f| f.borrow().get_state_factory_size())
}

pub fn add_search_tree_loop_transition(left: StateId, right: StateId, target: StateId) {
    STATE_FACTORY.with(|f| {
        f.borrow_mut()
            .add_search_tree_loop_transition(left, right, target)
    });
}

pub fn set_search_tree_top_level_transition(left: StateId, right: StateId, target: StateId) {
    STATE_FACTORY.with(|f| {
        f.borrow_mut()
            .set_search_tree_top_level_transition(left, right, target)
    });
}

pub fn remove_search_tree_loop_transition(left: StateId, right: StateId, target: StateId) {
    STATE_FACTORY.with(|f| {
        f.borrow_mut()
            .remove_search_tree_loop_transition(left, right, target)
    });
}

pub fn get_search_tree_transition(
    right: StateId,
) -> Option<(SearchTreeEntry, FxHashSet<SearchTreeEntry>)> {
    STATE_FACTORY.with(|f| f.borrow().get_search_tree_transition(right))
}

pub fn get_search_tree() -> FxHashMap<StateId, (SearchTreeEntry, FxHashSet<SearchTreeEntry>)> {
    STATE_FACTORY.with(|f| f.borrow().fta_search_tree.clone())
}

pub fn get_search_tree_size() -> usize {
    STATE_FACTORY.with(|f| {
        f.borrow()
            .fta_search_tree
            .values()
            .map(|(_, transitions)| transitions.len() + 1)
            .sum()
    })
}

pub fn set_is_same_action_type(
    p1: StateId,
    p2: StateId,
    is_same_action_type: bool,
    has_senddata: bool,
) {
    STATE_FACTORY.with(|f| {
        f.borrow_mut()
            .set_is_same_action_type(p1, p2, is_same_action_type, has_senddata)
    });
}

pub fn get_is_same_action_type(p1: StateId, p2: StateId) -> Option<(bool, bool)> {
    STATE_FACTORY.with(|f| f.borrow().get_is_same_action_type(p1, p2))
}

pub fn set_first_while_pattern(q0: StateId, pattern: Vec<ActionType>) {
    STATE_FACTORY.with(|f| f.borrow_mut().set_first_while_pattern(q0, pattern));
}

pub fn get_first_while_pattern(q0: StateId) -> Option<Vec<ActionType>> {
    STATE_FACTORY.with(|f| f.borrow().get_first_while_pattern(q0))
}

pub fn get_all_first_while_patterns() -> Vec<Vec<ActionType>> {
    STATE_FACTORY.with(|f| f.borrow().get_all_first_while_patterns())
}

pub fn set_while_body_last_stopped_dom(q_body: StateId, n: Index) {
    STATE_FACTORY.with(|f| f.borrow_mut().set_while_body_last_stopped_dom(q_body, n))
}

pub fn get_while_body_last_stopped_dom(q_body: StateId) -> Option<Index> {
    STATE_FACTORY.with(|f| f.borrow().get_while_body_last_stopped_dom(q_body))
}

pub fn set_for_body_last_stopped_dom(q_body: StateId, n: Index) {
    STATE_FACTORY.with(|f| f.borrow_mut().set_for_body_last_stopped_dom(q_body, n))
}

pub fn get_for_body_last_stopped_dom(q_body: StateId) -> Option<Index> {
    STATE_FACTORY.with(|f| f.borrow().get_for_body_last_stopped_dom(q_body))
}

pub fn set_is_early_stopped(stopped: bool) {
    STATE_FACTORY.with(|f| f.borrow_mut().set_is_early_stopped(stopped))
}

pub fn get_is_early_stopped() -> bool {
    STATE_FACTORY.with(|f| f.borrow().get_is_early_stopped())
}

thread_local! {
    pub static STATE_FACTORY: RefCell<StateFactory> = StateFactory::default().into();
}
