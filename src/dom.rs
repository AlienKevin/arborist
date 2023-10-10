use crate::selector::Index;
use scraper;
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

/// A list of `Dom`s in the same order as the `actions` in `ActionTrace`
///
/// Has exactly one more `Dom` than `Action` because of
/// the presence of an initial `Dom` before any action
/// is performed.
pub type Doms = Vec<Dom>;

/// A snapshot of the HTML document from the frontend
pub type Dom = scraper::Html;

/// Start index of the dom traces
/// End index is implicitly set to the length of doms vector
/// In the context of eval_selector(), DomTrace has a length of 1
/// and refers to the index of the dom trace in a vector of doms
#[derive(Clone, Copy, Debug)]
pub struct DomTrace {
    pub n: Index,
    pub number_of_doms: Index,
}

impl DomTrace {
    pub fn new(n: Index, number_of_doms: Index) -> Self {
        DomTrace { n, number_of_doms }
    }

    /// Check if the `DomTrace` is empty under `doms`
    pub fn is_empty(&self) -> bool {
        self.n >= self.number_of_doms
    }

    /// Check if the `DomTrace` is at the last dom under `doms`
    pub fn is_end(&self) -> bool {
        self.n == self.number_of_doms - 1
    }

    pub fn increment_number_of_doms(&self) -> Self {
        DomTrace {
            n: self.n,
            number_of_doms: self.number_of_doms + 1,
        }
    }

    pub fn increment_n_by(&self, m: Index) -> Self {
        DomTrace {
            n: self.n + m,
            number_of_doms: self.number_of_doms,
        }
    }

    pub fn increment_all(&self) -> Self {
        DomTrace {
            n: self.n + 1,
            number_of_doms: self.number_of_doms + 1,
        }
    }

    /// Advance the `DomTrace` to point to the next `Dom`
    pub fn next(&self) -> Self {
        DomTrace {
            n: self.n + 1,
            number_of_doms: self.number_of_doms,
        }
    }
}

impl PartialEq for DomTrace {
    fn eq(&self, other: &Self) -> bool {
        self.n == other.n
    }
}

impl PartialOrd for DomTrace {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.n.partial_cmp(&other.n)
    }
}

impl Eq for DomTrace {}

impl Ord for DomTrace {
    fn cmp(&self, other: &Self) -> Ordering {
        self.n.cmp(&other.n)
    }
}

impl Hash for DomTrace {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.n.hash(state);
    }
}

impl std::fmt::Display for DomTrace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "d{}", self.n)
    }
}

pub fn parse_html_to_dom(html: &str) -> Dom {
    scraper::Html::parse_document(html)
}
