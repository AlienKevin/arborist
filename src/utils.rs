use rustc_hash::FxHashSet;
use serde_json::{Map, Value};
use std::hash::Hash;

pub fn value_to_array(value: Value) -> Option<Vec<Value>> {
    match value {
        Value::Array(array) => Some(array),
        _ => None,
    }
}

pub fn value_to_object(value: Value) -> Option<Map<String, Value>> {
    match value {
        Value::Object(object) => Some(object),
        _ => None,
    }
}
