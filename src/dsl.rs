use crate::{selector::Selector, value_path::ValuePath};
use im_rc::Vector;
use std::str::FromStr;

/// A program is a sequence of `Expr`s
///
/// Program P ::= Seq(E, P) | Seq(E, Nil)
pub type Program = Vec<Expr>;

/// An expression
///
/// Expression E ::= Click(n) | For(ρ, N, P) | While(P, n)
#[derive(Clone)]
pub enum Expr {
    Nullary(NullaryOp),
    Unary(Vector<Selector>, UnaryOp),
    For(Box<Program>),
    While(Box<Program>, Vector<Selector>),
    SendKeys(Vector<Selector>, String),
    SendData(Vector<Selector>, ValuePath),
    ForData(ValuePath, Box<Program>),
}

#[derive(Hash, PartialEq, Ord, PartialOrd, Eq, Clone, Copy, Debug)]
pub enum NullaryOp {
    GoBack,
    ExtractURL,
}

impl std::fmt::Display for NullaryOp {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            NullaryOp::GoBack => write!(f, "GoBack"),
            NullaryOp::ExtractURL => write!(f, "ExtractURL"),
        }
    }
}

impl FromStr for NullaryOp {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "GoBack" => Ok(NullaryOp::GoBack),
            "ExtractURL" => Ok(NullaryOp::ExtractURL),
            _ => Err("Unrecognized NullaryOp".to_string()),
        }
    }
}

#[derive(Hash, PartialEq, Ord, PartialOrd, Eq, Clone, Copy, Debug)]
pub enum UnaryOp {
    Click,
    ScrapeText,
    Download,
    ScrapeLink,
}

impl std::fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            UnaryOp::Click => write!(f, "Click"),
            UnaryOp::ScrapeText => write!(f, "ScrapeText"),
            UnaryOp::Download => write!(f, "Download"),
            UnaryOp::ScrapeLink => write!(f, "ScrapeLink"),
        }
    }
}

impl FromStr for UnaryOp {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Click" => Ok(UnaryOp::Click),
            "ScrapeText" => Ok(UnaryOp::ScrapeText),
            "Download" => Ok(UnaryOp::Download),
            "ScrapeLink" => Ok(UnaryOp::ScrapeLink),
            _ => Err("Unrecognized UnaryOp".to_string()),
        }
    }
}

/// A list of Programs each with alternative subprograms
pub type GroupedPrograms = Vec<GroupedProgram>;

/// A grouped Program with alternative subprograms
/// that might have the same or different behaviors
#[derive(Clone, Debug)]
pub enum GroupedProgram {
    GroupedSeq(Vec<GroupedExpr>, Vec<Box<GroupedProgram>>),
    GroupedSingle(Vec<GroupedExpr>),
}

pub fn program_to_string(program: &Program, indent: &str) -> String {
    let mut rlt = "".to_string();
    for expr in program {
        match expr {
            Expr::Nullary(op) => rlt.push_str(&format!("\n{indent}{op}")),
            Expr::Unary(s, op) => rlt.push_str(&format!("\n{indent}{op} {}", s[0])),
            Expr::SendKeys(s, k) => rlt.push_str(&format!("\n{indent}SendKeys {} {k}", s[0])),
            Expr::SendData(s, v) => rlt.push_str(&format!("\n{indent}SendData {} {v}", s[0])),
            Expr::For(p) => {
                rlt.push_str(&format!("\n{}ForEach v in", indent));
                rlt.push_str(&program_to_string(
                    p,
                    &format!("{}", indent.to_owned() + "    "),
                ));
            }
            Expr::While(p, ns) => {
                rlt.push_str(&format!("\n{}Do", indent,));
                rlt.push_str(&program_to_string(
                    p,
                    &format!("{}", indent.to_owned() + "    "),
                ));
                rlt.push_str(&format!("\n{}While {}", indent, ns[0]));
            }
            Expr::ForData(v, p) => {
                rlt.push_str(&format!(
                    "\n{}ForEach v in value_path: {}",
                    indent,
                    v.to_string()
                ));
                rlt.push_str(&program_to_string(
                    p,
                    &format!("{}", indent.to_owned() + "    "),
                ));
            }
        }
    }
    rlt
}

impl From<Vec<GroupedExpr>> for GroupedProgram {
    fn from(exprs: Vec<GroupedExpr>) -> Self {
        match &exprs[..] {
            [] => panic!("Input GroupedExprs is empty"),
            [_] => GroupedProgram::GroupedSingle(exprs),
            [head, tail @ ..] => GroupedProgram::GroupedSeq(
                vec![head.clone()],
                vec![Box::new(Self::from(tail.to_vec()))],
            ),
        }
    }
}

/// A grouped Expr with alternative subexpressions
/// that might have the same or different behaviors
#[derive(Clone, Debug)]
pub enum GroupedExpr {
    Nullary(NullaryOp),
    Unary(Vector<Selector>, UnaryOp),
    For(Vec<Box<GroupedProgram>>),
    While(Vec<Box<GroupedProgram>>, Vector<Selector>),
    SendKeys(Vector<Selector>, String),
    SendData(Vector<Selector>, ValuePath),
    ForData(ValuePath, Vec<Box<GroupedProgram>>),
}
