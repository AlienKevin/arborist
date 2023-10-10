use std::str::FromStr;

use serde_json::Value;

/// An abstract selector that refers to a node in `Dom`
///
/// Value Path 𝑣 ::=
///     | x (Empty)
///     | 𝜗 (Var)
///     | 𝑣[key] (key)
///     | 𝑣[i] (index)
///
/// `Pred` is added by Kevin for convenience during testing

#[derive(Hash, Ord, PartialOrd, PartialEq, Eq, Clone, Debug)]
pub enum Index {
    Key(String),
    Number(usize),
}

#[derive(Hash, PartialEq, Ord, PartialOrd, Eq, Clone, Debug)]
pub struct ValuePath(pub Vec<Index>);

impl ValuePath {
    pub fn new() -> ValuePath {
        ValuePath(Vec::new())
    }

    pub fn from_path(path: Vec<Index>) -> ValuePath {
        ValuePath(path)
    }

    pub fn append_index(&self, index: Index) -> ValuePath {
        let mut new_path = self.0.clone();
        new_path.push(index);
        ValuePath(new_path)
    }

    pub fn append(&self, value_path: &ValuePath) -> ValuePath {
        let mut new_path = self.0.clone();
        new_path.extend(value_path.0.clone());
        ValuePath(new_path)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, index: usize) -> Option<&Index> {
        self.0.get(index)
    }

    pub fn last(&self) -> Index {
        self.0.last().unwrap().clone()
    }
}

impl Default for ValuePath {
    fn default() -> Self {
        Self::new()
    }
}

pub fn value_path_to_string(paths: &ValuePath) -> String {
    if paths.0.is_empty() {
        "Empty".to_string()
    } else {
        let mut s = "".to_string();
        for key in &paths.0 {
            match key {
                Index::Key(key) => s.push_str(&format!("[{}]", key)),
                Index::Number(number) => s.push_str(&format!("[{}]", number)),
            }
        }
        s
    }
}

impl std::fmt::Display for ValuePath {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", value_path_to_string(self))
    }
}

impl FromStr for ValuePath {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(parse_value_path(s))
    }
}

fn parse_value_path(value_path: &str) -> ValuePath {
    let mut path = Vec::new();
    let mut current_index = 0;
    let chars: Vec<char> = value_path.chars().collect();

    while current_index < chars.len() {
        if chars[current_index] == '[' {
            // Move current_index past the opening bracket
            current_index += 1;

            // Check if the next character is a number
            if chars[current_index].is_numeric() {
                // Parse the number until the closing bracket
                let mut number = String::new();
                while current_index < chars.len() && chars[current_index].is_numeric() {
                    number.push(chars[current_index]);
                    current_index += 1;
                }
                // Convert the parsed number into usize and add to the path
                if let Ok(index) = number.parse::<usize>() {
                    path.push(Index::Number(index));
                } else {
                    panic!("Failed to parse number in value path");
                }
            } else {
                // Parse the string until the closing bracket
                let mut string = String::new();
                while current_index < chars.len() && chars[current_index] != ']' {
                    string.push(chars[current_index]);
                    current_index += 1;
                }
                // Add the parsed string as a key to the path
                path.push(Index::Key(string));
            }

            // Move current_index past the closing bracket
            current_index += 1;
        } else if chars[current_index].is_alphabetic()
            || chars[current_index].is_numeric()
            || chars[current_index] == '.'
        {
            if chars[current_index] == '.' {
                current_index += 1;
            }
            // Parse the key or number until the next delimiter
            let mut key_or_number = String::new();
            while current_index < chars.len()
                && chars[current_index] != '['
                && chars[current_index] != ']'
                && chars[current_index] != '.'
            {
                key_or_number.push(chars[current_index]);
                current_index += 1;
            }
            // Try to convert the parsed key or number into usize
            if let Ok(index) = key_or_number.parse::<usize>() {
                path.push(Index::Number(index));
            } else {
                // Add the parsed key as a string to the path
                path.push(Index::Key(key_or_number));
            }
        } else {
            panic!("Invalid character in value path");
        }

        // Skip any additional delimiter characters
        while current_index < chars.len()
            && (chars[current_index] == '[' || chars[current_index] == ']')
        {
            current_index += 1;
        }
    }

    ValuePath(path)
}

pub fn anti_unify_paths(paths1: &ValuePath, paths2: &ValuePath) -> Option<ValuePath> {
    let mut new_paths = Vec::new();
    if paths1.len() != paths2.len() {
        return None;
    }
    // if paths1.last() != Index::Number(0) || paths2.last() != Index::Number(1) {
    //     return None;
    // }
    for (index1, index2) in paths1.0.iter().zip(paths2.0.iter()) {
        match (index1, index2) {
            (Index::Key(key1), Index::Key(key2)) => {
                if *key1 == *key2 {
                    new_paths.push(Index::Key(key1.clone()));
                } else {
                    return None;
                }
            }
            (Index::Number(number1), Index::Number(number2)) => {
                if *number1 == 0 && *number2 == 1 {
                    break;
                }
                if number1 == number2 {
                    new_paths.push(Index::Number(*number1));
                } else {
                    return None;
                }
            }
            _ => {
                panic!("Invalid index in value path")
            }
        }
    }
    Some(ValuePath(new_paths))
}

pub fn get_list_at_path(data: &Value, path: &Vec<Index>) -> Option<Vec<Value>> {
    if path.is_empty() {
        return if data.is_array() {
            data.as_array().cloned()
        } else {
            None
        };
    }
    match &path[0] {
        Index::Key(key) => match data.get(key) {
            Some(val) => get_list_at_path(val, &path[1..].to_vec()),
            None => None,
        },
        Index::Number(key) => match data.get(key) {
            Some(val) => get_list_at_path(val, &path[1..].to_vec()),
            None => None,
        },
    }
}
