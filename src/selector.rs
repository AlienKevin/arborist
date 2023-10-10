use super::{dom::Dom, env};
use im_rc::{vector, Vector};
use internment::Intern;
use itertools::{Either, Itertools};
use regex::Regex;
use smallvec::{smallvec, SmallVec};
use std::slice::Iter;
use std::str::FromStr;
use ustr::{ustr, Ustr};

// TODO: be careful with potential overflow of index under u16
// currently not likely as the indices max out under 3000
pub type Index = u16;

pub type Indices = SmallVec<[Index; 4]>;

/// An abstract selector that refers to a node in `Dom`
///
/// Selector n ::=
///     | ε (Empty)
///     | ρ (Var)
///     | n/ɸ[i] (Child)
///     | n//ɸ[i] (Descendant)
///
/// `Pred` is added by Kevin for convenience during testing
#[derive(Hash, Ord, PartialOrd, PartialEq, Eq, Clone)]
pub enum Selector {
    Abstract(SelectorPath, Indices), // The last index of the SelectorPath inside SmallVec is the start_index of the prefix
    Concrete(SelectorPath),
}

pub type SelectorSegment = (Relationship, Pred);
pub type SelectorSegmentsRaw = SmallVec<[SelectorSegment; 2]>;
pub type SelectorSegments = Intern<SelectorSegmentsRaw>;

#[derive(Hash, Ord, PartialOrd, PartialEq, Eq, Clone, Default, Debug)]
pub struct SelectorPath {
    pub path: SelectorSegments,
    pub indices: Indices,
}

impl SelectorPath {
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    pub fn len(&self) -> usize {
        self.indices.len()
    }
}

pub struct SelectorPathIter<'a> {
    path_iter: Iter<'a, SelectorSegment>,
    indices_iter: Iter<'a, Index>,
}

impl<'a> IntoIterator for &'a SelectorPath {
    type Item = ((Relationship, Pred), Index);
    type IntoIter = SelectorPathIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        SelectorPathIter {
            path_iter: self.path.iter(),
            indices_iter: self.indices.iter(),
        }
    }
}

impl<'a> Iterator for SelectorPathIter<'a> {
    type Item = ((Relationship, Pred), Index);

    fn next(&mut self) -> Option<Self::Item> {
        match (self.path_iter.next(), self.indices_iter.next()) {
            (Some((rel, pred)), Some(index)) => Some(((*rel, pred.clone()), *index)),
            _ => None,
        }
    }
}

#[derive(Hash, Ord, PartialOrd, PartialEq, Eq, Clone, Default, Debug)]
pub struct SelectorPathRaw {
    pub path: SelectorSegmentsRaw,
    pub indices: Indices,
}

impl SelectorPathRaw {
    pub fn intern(self) -> SelectorPath {
        SelectorPath {
            path: Intern::new(self.path),
            indices: self.indices,
        }
    }

    pub fn len(&self) -> usize {
        self.indices.len()
    }
}

impl<'a> IntoIterator for &'a SelectorPathRaw {
    type Item = ((Relationship, Pred), Index);
    type IntoIter = SelectorPathIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        SelectorPathIter {
            path_iter: self.path.iter(),
            indices_iter: self.indices.iter(),
        }
    }
}

fn path_to_string(
    path: &SelectorSegmentsRaw,
    indices: &Indices,
    show_empty: PathShowEmpty,
) -> String {
    if show_empty == PathShowEmpty::Show && path.is_empty() {
        "Empty".to_string()
    } else {
        path.iter()
            .zip(indices)
            .map(|((rel, pred), index)| {
                let rel_str = match rel {
                    Relationship::Child => "/",
                    Relationship::Descendant => "//",
                };
                format!("{}{}[{}]", rel_str, pred, index)
            })
            .join("")
    }
}

fn abstract_selector_to_string(
    path: &SelectorPath,
    indices: &Indices,
    show_empty: PathShowEmpty,
) -> String {
    if show_empty == PathShowEmpty::Show && path.is_empty() {
        "Empty".to_string()
    } else {
        path.path
            .iter()
            .zip(&path.indices)
            .enumerate()
            .map(|(i, ((rel, pred), index))| {
                let rel_str = match rel {
                    Relationship::Child => "/",
                    Relationship::Descendant => "//",
                };
                if indices.contains(&(i as Index)) {
                    format!("{}{}[[{}]]", rel_str, pred, index)
                } else {
                    format!("{}{}[{}]", rel_str, pred, index)
                }
            })
            .join("")
    }
}

#[derive(Hash, Ord, PartialOrd, PartialEq, Eq, Clone, Copy, Debug)]
pub enum Relationship {
    Child,
    Descendant,
}

#[derive(PartialEq)]
enum PathShowEmpty {
    Show,
    Hide,
}

impl Selector {
    /// Convert an abstract selector to concrete
    /// by looking up concrete values for the variable
    /// in the `context`
    ///
    /// Σ ⊢ n ⇝ ρ
    pub fn to_concrete_selector(&self, context: &env::Env) -> ConcreteSelector {
        match self {
            Self::Abstract(
                SelectorPath {
                    path,
                    indices: path_indices,
                },
                breakpoints,
            ) => {
                assert!(context.is_some());
                let indices = context.as_ref().unwrap();
                assert!(indices.len() >= breakpoints.len());
                let mut path_indices = path_indices.clone();
                for (&breakpoint, &i) in breakpoints
                    .iter()
                    .zip(&indices[indices.len() - breakpoints.len()..])
                {
                    path_indices[breakpoint as usize] += i - 1;
                }
                ConcreteSelector::Path(SelectorPath {
                    path: *path,
                    indices: path_indices,
                })
            }
            Self::Concrete(path) => ConcreteSelector::Path(path.clone()),
        }
    }

    pub fn is_abstract(&self) -> bool {
        match self {
            Self::Abstract(..) => true,
            Self::Concrete(_) => false,
        }
    }

    pub fn parametrize(
        ns1: Vector<Selector>,
        ns2: Vector<Selector>,
        depth: Index,
        max_big_n_start_index: Index,
    ) -> (bool, Vector<Selector>) {
        let (abstract1, concrete1): (Vec<_>, Vec<_>) = ns1.into_iter().partition_map(|n| match n {
            Self::Abstract(prefixes, path) => Either::Left((prefixes, path)),
            Self::Concrete(path) => Either::Right(path),
        });
        let (abstract2, concrete2): (Vec<_>, Vec<_>) = ns2.into_iter().partition_map(|n| match n {
            Self::Abstract(prefixes, path) => Either::Left((prefixes, path)),
            Self::Concrete(path) => Either::Right(path),
        });
        let mut results = vector![];
        if depth == 0 {
            assert!(abstract1.is_empty());
            assert!(abstract2.is_empty());
            for path1 in &concrete1 {
                for path2 in &concrete2 {
                    if path1.path == path2.path {
                        if let Some(breakpoint) = anti_unify_paths(
                            &path1.indices,
                            &path2.indices,
                            path1.len() as Index,
                            max_big_n_start_index,
                        ) {
                            results
                                .push_back(Selector::Abstract(path1.clone(), smallvec![breakpoint]))
                        }
                    }
                }
            }
            let anti_unified = !results.is_empty();
            let concrete_intersects = find_intersection(concrete1, &concrete2);
            results.append(concrete_intersects);
            (anti_unified, results)
        } else {
            assert!(abstract1
                .iter()
                .all(|(_, indices)| indices.len() <= depth as usize));
            assert!(abstract2
                .iter()
                .all(|(_, indices)| indices.len() <= depth as usize));
            for (
                SelectorPath {
                    path: path1,
                    indices: indices1,
                },
                breakpoints1,
            ) in &abstract1
            {
                for (
                    SelectorPath {
                        path: path2,
                        indices: indices2,
                    },
                    breakpoints2,
                ) in &abstract2
                {
                    if breakpoints1.len() == depth as usize
                        && path1 == path2
                        && breakpoints1 == breakpoints2
                        && indices1[breakpoints1[0] as usize..]
                            == indices2[breakpoints1[0] as usize..]
                    {
                        if let Some(breakpoint) = anti_unify_paths(
                            indices1,
                            indices2,
                            breakpoints1[0],
                            max_big_n_start_index,
                        ) {
                            let mut breakpoints = smallvec![breakpoint];
                            breakpoints.extend_from_slice(breakpoints1);
                            results.push_back(Selector::Abstract(
                                SelectorPath {
                                    path: *path1,
                                    indices: indices1.clone(),
                                },
                                breakpoints,
                            ))
                        }
                    }
                }
            }
            let anti_unified = !results.is_empty();
            results.append(find_intersection(concrete1, &concrete2));
            results.append(find_abstract_intersection(abstract1, &abstract2));
            (anti_unified, results)
        }
    }
}

fn find_abstract_intersection(
    ns1: Vec<(SelectorPath, Indices)>,
    ns2: &Vec<(SelectorPath, Indices)>,
) -> Vector<Selector> {
    let mut intersection = Vector::new();

    for n1 in ns1 {
        if ns2.contains(&n1) {
            intersection.push_back(Selector::Abstract(n1.0, n1.1));
        }
    }

    intersection
}

fn find_intersection(ns1: Vec<SelectorPath>, ns2: &Vec<SelectorPath>) -> Vector<Selector> {
    let mut intersection = Vector::new();

    for item in ns1 {
        if ns2.contains(&item) {
            intersection.push_back(Selector::Concrete(item));
        }
    }

    intersection
}

pub fn find_intersection_selectors(
    ns1: Vector<Selector>,
    ns2: &Vector<Selector>,
) -> Vector<Selector> {
    let mut intersection = Vector::new();

    for item in ns1 {
        if ns2.contains(&item) {
            intersection.push_back(item);
        }
    }

    intersection
}

impl std::fmt::Display for Selector {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Abstract(path, indices) => {
                write!(
                    f,
                    "v{}",
                    abstract_selector_to_string(path, indices, PathShowEmpty::Hide)
                )
            }
            Self::Concrete(path) => {
                write!(
                    f,
                    "{}",
                    path_to_string(&path.path, &path.indices, PathShowEmpty::Show)
                )
            }
        }
    }
}

impl std::fmt::Debug for Selector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

/// A concrete selector without any variables
///
/// ρ ::=
///     | ε (Empty)
///     | ρ/ɸ[i] (Child)
///     | ρ//ɸ[i] (Descendant)
///
/// `Pred` is added by Kevin for convenience during testing.
/// `NodeId` represents an absolute selector created by calling `to_abstract_selector`.
#[derive(Hash, PartialOrd, Ord, PartialEq, Eq, Clone)]
pub enum ConcreteSelector {
    Path(SelectorPath),
    NodeId(scraper::ego_tree::NodeId),
    MatrixIndex(Index),
}

/// If [i] is not present after ɸ, i is assumed to be 1
impl FromStr for ConcreteSelector {
    type Err = Box<dyn std::error::Error>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::path_from_str(s).map(|s| Self::Path(s.intern()))
    }
}

impl ConcreteSelector {
    pub fn empty() -> Self {
        Self::Path(SelectorPath::default())
    }

    pub fn as_path(&self) -> &SelectorPath {
        match self {
            ConcreteSelector::Path(path) => path,
            ConcreteSelector::NodeId(_) => {
                panic!("ConcreteSelector::as_path() does not support NodeId")
            }
            ConcreteSelector::MatrixIndex(_) => {
                panic!("ConcreteSelector::as_path() does not support MatrixIndex")
            }
        }
    }

    pub fn path_from_str(s: &str) -> Result<SelectorPathRaw, Box<dyn std::error::Error>> {
        if s.is_empty() {
            Ok(SelectorPathRaw::default())
        } else {
            let mut path = smallvec![];
            let mut indices = smallvec![];
            let chars = s.chars().collect::<Vec<char>>();
            let mut i = chars.len() - 1;

            loop {
                // get the index if there is any
                // defaults to 1 if none is specified
                let mut index = 1;
                // TODO: (perf) optimize into_iter().collect for is_match
                // Consider using pointers to chars instead
                if chars[i] == ']' && chars[i - 1] != '\'' {
                    let index_end = i;
                    while chars[i] != '[' {
                        i -= 1;
                    }
                    index = chars[i + 1..index_end]
                        .iter()
                        .collect::<String>()
                        .parse::<Index>()?;
                    // skip the opening bracket '['
                    i -= 1;
                }
                let pred_end = i;
                if chars[i] == ']' {
                    // Handle edge case where '/' is in the attribute value
                    // eg: //div[@data-value='09/30/2021']
                    // Skip the closing bracket and single quote
                    i -= 2;
                    while chars[i] != '\'' {
                        i -= 1;
                    }
                }
                while chars[i] != '/' {
                    i -= 1;
                }
                // i points to the second '/' if it is a descendant or simply '/' if it is a child
                let pred = chars[i + 1..pred_end + 1]
                    .iter()
                    .collect::<String>()
                    .parse::<Pred>()?;
                let mut reached_the_start = false;
                // Found a descendant selector
                let relationship = if i >= 1 && chars[i - 1] == '/' {
                    if i <= 1 {
                        reached_the_start = true;
                    } else {
                        i -= 2; // skip the '//'
                    }
                    Relationship::Descendant
                }
                // Found a child selector
                else {
                    if i == 0 {
                        reached_the_start = true;
                    } else {
                        i -= 1; // skip the '/'
                    }
                    Relationship::Child
                };
                path.push((relationship, pred));
                indices.push(index);
                if reached_the_start {
                    break;
                }
            }
            path.reverse();
            indices.reverse();
            Ok(SelectorPathRaw { path, indices })
        }
    }

    /// Convert a concrete selector to an abstract selector
    ///
    /// Occasionally useful in testing.
    /// CANNOT handle ConcreteSelector::NodeId
    pub fn to_abstract_selector(&self) -> Selector {
        match self {
            Self::Path(path) => Selector::Concrete(path.clone()),
            Self::NodeId(_) => {
                panic!("NodeId cannot be convert to an abstract selector");
            }
            Self::MatrixIndex(_) => {
                panic!("MatrixIndex cannot be convert to an abstract selector");
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        matches!(self, Self::Path(path) if path.is_empty())
    }

    pub fn to_query_xpath(&self) -> String {
        match self {
            Self::Path(SelectorPath { path, indices }) => {
                if path.is_empty() {
                    panic!("to_query_xpath took in an empty selector")
                } else {
                    path.iter()
                        .zip(indices)
                        .fold("".to_string(), |result, ((rel, pred), index)| {
                            let rel_str = match rel {
                                Relationship::Child => "/",
                                Relationship::Descendant => "//",
                            };
                            format!("({result}{rel_str}{pred})[{index}]")
                        })
                }
            }
            Self::NodeId(_) => {
                panic!("to_query_xpath does not support NodeId")
            }
            Self::MatrixIndex(_) => {
                panic!("to_query_xpath does not support MatrixIndex")
            }
        }
    }
}

impl std::fmt::Display for ConcreteSelector {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Path(path) => {
                write!(
                    f,
                    "{}",
                    path_to_string(&path.path, &path.indices, PathShowEmpty::Show)
                )
            }
            Self::NodeId(_) => {
                write!(f, "NodeId")
            }
            Self::MatrixIndex(_) => {
                write!(f, "MatrixIndex")
            }
        }
    }
}

impl std::fmt::Debug for ConcreteSelector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

// Click(a//b[1]//c/d)
// Click(a//b[2]//c/d)
// N = a//b
// Click(v//c/d) where v = N[i]
// postfixes = [v, //c, /d]
fn anti_unify_paths(
    indices1: &Indices,
    indices2: &Indices,
    end_index: Index,
    max_big_n_start_index: Index,
) -> Option<Index> {
    let mut i = 0;
    while i < end_index {
        let i1 = indices1[i as usize];
        let i2 = indices2[i as usize];
        if i1 <= max_big_n_start_index && i2 == i1 + 1 {
            break;
        } else if i1 != i2 {
            return None;
        }
        i += 1;
    }
    if i >= end_index {
        None
    } else {
        let prediction_end = i;
        i += 1;
        while i < end_index {
            if indices1[i as usize] != indices2[i as usize] {
                return None;
            }
            i += 1;
        }
        Some(prediction_end)
    }
}

/// A predicate used in selectors
///
/// Predicate ɸ ::= tag | tag[@t=s]
#[derive(Hash, Ord, PartialOrd, PartialEq, Eq, Clone, Debug)]
pub struct Pred {
    pub tag: Ustr,
    pub attr: Option<Attr>,
}

impl FromStr for Pred {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.split_once('[') {
            Some((tag, attr)) => Ok(Pred {
                tag: ustr(tag),
                attr: Some(Attr::from_str(&('['.to_string() + attr)).unwrap()),
            }),
            None => Ok(Pred {
                tag: ustr(s),
                attr: None,
            }),
        }
    }
}

impl std::fmt::Display for Pred {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{}{}",
            self.tag,
            self.attr
                .as_ref()
                .map(|attr| attr.to_string())
                .unwrap_or_default()
        )
    }
}

/// An HTML attribute used in `Pred`
/// Attribute ::= [@t=s]
/// For example: [@class='link btn'], [@id='main'] , [@data-sqe='name']
#[derive(Hash, Ord, PartialOrd, PartialEq, Eq, Clone, Debug)]
pub enum Attr {
    Class(Ustr),
    Other(Ustr, Ustr),
    Id(Ustr),
}

lazy_static::lazy_static! {
    static ref ATTR_REGEX: Regex = Regex::new(r"\[@?(?P<t>.+)='(?P<s>[^'\\]*(\\\\|\\n|[^'\\])*)'\]"
).unwrap();
}
// original  r"\[@(?P<t>.+)='\s*(?P<s>.+)\s*'\]"

impl FromStr for Attr {
    type Err = String;

    /// Captures t and s in [@t=s]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let captures = ATTR_REGEX
            .captures(s)
            .ok_or_else(|| format!("Invalid attribute: {}", s))?;
        let t = captures.name("t").unwrap().as_str();
        let s = captures.name("s").unwrap().as_str();
        match t {
            "class" => Ok(Attr::Class(ustr(s))),
            "id" => Ok(Attr::Id(ustr(s))),
            t => Ok(Attr::Other(ustr(t), ustr(s))),
        }
    }
}

impl std::fmt::Display for Attr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Class(class) => {
                // NOTE: Multiple class names may be separated by whitespace
                write!(f, "[@class='{}']", class)
            }
            Self::Other(name, value) => {
                write!(f, "[@{}='{}']", name, value)
            }
            Self::Id(id) => {
                write!(f, "[@id='{}']", id)
            }
        }
    }
}

/*
#[test]
fn test_path_from_str() {
    {
        let n = "//div[@data-value='09/30/2021']/span[@class='a']";
        let expected: SelectorSegmentsRaw = smallvec![
            (
                Relationship::Descendant,
                Pred {
                    tag: ustr("div"),
                    attr: Some(Attr::Other(ustr("data-value"), ustr("09/30/2021"),))
                },
                1
            ),
            (
                Relationship::Child,
                Pred {
                    tag: ustr("span"),
                    attr: Some(Attr::Classes(SortedUstrVec::from(vec![ustr("a")])))
                },
                1
            )
        ];
        assert_eq!(ConcreteSelector::path_from_str(n).unwrap(), expected)
    }
    {
        let n = "//div[@data-value='09/30/2021']/span[@class='a'][2]";
        let expected: SelectorSegmentsRaw = smallvec![
            (
                Relationship::Descendant,
                Pred {
                    tag: ustr("div"),
                    attr: Some(Attr::Other(ustr("data-value"), ustr("09/30/2021"),))
                },
                1
            ),
            (
                Relationship::Child,
                Pred {
                    tag: ustr("span"),
                    attr: Some(Attr::Classes(SortedUstrVec::from(vec![ustr("a")])))
                },
                2
            )
        ];
        assert_eq!(ConcreteSelector::path_from_str(n).unwrap(), expected)
    }
    {
        let n = "//div[@data-value='09/30/2021'][2]/span[@class='a']";
        let expected: SelectorSegmentsRaw = smallvec![
            (
                Relationship::Descendant,
                Pred {
                    tag: ustr("div"),
                    attr: Some(Attr::Other(ustr("data-value"), ustr("09/30/2021"),))
                },
                2
            ),
            (
                Relationship::Child,
                Pred {
                    tag: ustr("span"),
                    attr: Some(Attr::Classes(SortedUstrVec::from(vec![ustr("a")])))
                },
                1
            )
        ];
        assert_eq!(ConcreteSelector::path_from_str(n).unwrap(), expected)
    }
    {
        let n = "//div[@data-value='09/30/2021'][2]/span[@class='a'][3]";
        let expected: SelectorSegmentsRaw = smallvec![
            (
                Relationship::Descendant,
                Pred {
                    tag: ustr("div"),
                    attr: Some(Attr::Other(ustr("data-value"), ustr("09/30/2021"),))
                },
                2
            ),
            (
                Relationship::Child,
                Pred {
                    tag: ustr("span"),
                    attr: Some(Attr::Classes(SortedUstrVec::from(vec![ustr("a")])))
                },
                3
            )
        ];
        assert_eq!(ConcreteSelector::path_from_str(n).unwrap(), expected)
    }
    {
        let n = "//div[@data-value='09/30/2021'][2]/span";
        let expected: SelectorSegmentsRaw = smallvec![
            (
                Relationship::Descendant,
                Pred {
                    tag: ustr("div"),
                    attr: Some(Attr::Other(ustr("data-value"), ustr("09/30/2021"),))
                },
                2
            ),
            (
                Relationship::Child,
                Pred {
                    tag: ustr("span"),
                    attr: None,
                },
                1
            )
        ];
        assert_eq!(ConcreteSelector::path_from_str(n).unwrap(), expected)
    }
    {
        let n = "//div[2]/span";
        let expected: SelectorSegmentsRaw = smallvec![
            (
                Relationship::Descendant,
                Pred {
                    tag: ustr("div"),
                    attr: None,
                },
                2
            ),
            (
                Relationship::Child,
                Pred {
                    tag: ustr("span"),
                    attr: None,
                },
                1
            )
        ];
        assert_eq!(ConcreteSelector::path_from_str(n).unwrap(), expected)
    }
    {
        let n = "//div//span";
        let expected: SelectorSegmentsRaw = smallvec![
            (
                Relationship::Descendant,
                Pred {
                    tag: ustr("div"),
                    attr: None,
                },
                1
            ),
            (
                Relationship::Descendant,
                Pred {
                    tag: ustr("span"),
                    attr: None,
                },
                1
            )
        ];
        assert_eq!(ConcreteSelector::path_from_str(n).unwrap(), expected)
    }
    {
        let n = "//div[@class='w-2/3 md:w-3/4 flex flex-row justify-center'][1]";
        assert_eq!(
            ConcreteSelector::path_from_str(n).unwrap()[0],
            (
                Relationship::Descendant,
                Pred {
                    tag: ustr("div"),
                    attr: Some(Attr::Classes(SortedUstrVec::from(vec![
                        ustr("w-2/3"),
                        ustr("md:w-3/4"),
                        ustr("flex"),
                        ustr("flex-row"),
                        ustr("justify-center"),
                    ])))
                },
                1
            )
        )
    }
    {
        let n = "//div[@data-value='09/30/2021']";
        assert_eq!(
            ConcreteSelector::path_from_str(n).unwrap()[0],
            (
                Relationship::Descendant,
                Pred {
                    tag: ustr("div"),
                    attr: Some(Attr::Other(ustr("data-value"), ustr("09/30/2021"),))
                },
                1
            )
        )
    }
    {
        let n = "//div[@data-value='09/30/2021'][2]";
        assert_eq!(
            ConcreteSelector::path_from_str(n).unwrap()[0],
            (
                Relationship::Descendant,
                Pred {
                    tag: ustr("div"),
                    attr: Some(Attr::Other(ustr("data-value"), ustr("09/30/2021"),))
                },
                2
            )
        )
    }
    {
        let n = "//div[@data-component-name='\"Settlements\"']";
        assert_eq!(
            ConcreteSelector::path_from_str(n).unwrap()[0],
            (
                Relationship::Descendant,
                Pred {
                    tag: ustr("div"),
                    attr: Some(Attr::Other(
                        ustr("data-component-name"),
                        ustr("\"Settlements\""),
                    ))
                },
                1
            )
        )
    }
    {
        let n = "//div";
        assert_eq!(
            ConcreteSelector::path_from_str(n).unwrap()[0],
            (
                Relationship::Descendant,
                Pred {
                    tag: ustr("div"),
                    attr: None
                },
                1
            )
        )
    }
    {
        let n = "//div[2]";
        assert_eq!(
            ConcreteSelector::path_from_str(n).unwrap()[0],
            (
                Relationship::Descendant,
                Pred {
                    tag: ustr("div"),
                    attr: None
                },
                2
            )
        )
    }
    {
        let n = "//span[text()='Next']";
        assert_eq!(
            ConcreteSelector::path_from_str(n).unwrap()[0],
            (
                Relationship::Descendant,
                Pred {
                    tag: ustr("span"),
                    attr: Some(Attr::Other(ustr("text()"), ustr("Next"),))
                },
                1
            )
        )
    }
}
*/
