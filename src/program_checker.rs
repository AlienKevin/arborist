use itertools::Itertools;
use pest::{iterators::Pairs, Parser};

use crate::dsl::{NullaryOp, Program, UnaryOp};

mod old {
    use pest_derive::Parser;

    #[derive(Parser)]
    #[grammar = "ground_truth_old.pest"]
    pub struct GTParser;
}

mod new {
    use pest_derive::Parser;

    #[derive(Parser)]
    #[grammar = "ground_truth_new.pest"]
    pub struct GTParser;
}

#[derive(PartialEq, Eq, Debug)]
pub enum GTProgram {
    P(Vec<GTStatement>),
    NDSL,
    NGT,
}

#[derive(PartialEq, Eq, Debug)]
pub enum GTStatement {
    Nullary(NullaryOp),
    Unary(UnaryOp),
    For(GTProgram),
    While(GTProgram),
    SendKeys,
    SendData,
}

impl From<Program> for GTProgram {
    fn from(statements: Program) -> Self {
        use crate::dsl::Expr::*;
        GTProgram::P(
            statements
                .iter()
                .map(|statement| match statement {
                    &Nullary(op) => GTStatement::Nullary(op),
                    &Unary(_, op) => GTStatement::Unary(op),
                    For(body) => GTStatement::For(GTProgram::from(*body.clone())),
                    While(body, _) => GTStatement::While(GTProgram::from(*body.clone())),
                    SendKeys(..) => GTStatement::SendKeys,
                    SendData(..) => GTStatement::SendData,
                    ForData(_, body) => GTStatement::For(GTProgram::from(*body.clone())),
                })
                .collect_vec(),
        )
    }
}

pub fn get_program_stats(program: &GTProgram) -> (usize, usize, usize) {
    let mut max_depth = 0;
    let mut n_loop = 0;
    let mut n_ast = 0;
    if let GTProgram::P(statements) = program {
        for statement in statements {
            match statement {
                GTStatement::Nullary(_) => n_ast += 1,
                GTStatement::Unary(_) => n_ast += 1,
                GTStatement::For(body) => {
                    let (depth, n_loop_, n_ast_) = get_program_stats(body);
                    max_depth = max_depth.max(depth + 1);
                    n_loop += n_loop_ + 1;
                    n_ast += n_ast_ + 1;
                }
                GTStatement::While(body) => {
                    let (depth, n_loop_, n_ast_) = get_program_stats(body);
                    max_depth = max_depth.max(depth + 1);
                    n_loop += n_loop_ + 1;
                    n_ast += n_ast_ + 1;
                }
                GTStatement::SendKeys => n_ast += 1,
                GTStatement::SendData => n_ast += 1,
            }
        }
    }
    (max_depth, n_loop, n_ast)
}

impl From<Pairs<'_, old::Rule>> for GTProgram {
    fn from(value: Pairs<old::Rule>) -> Self {
        let statements = value
            .filter_map(|pair| match pair.as_rule() {
                old::Rule::nullary_statement => {
                    let nullary_op = match pair.as_str() {
                        "ExtractURL" => NullaryOp::ExtractURL,
                        "GoBack" => NullaryOp::GoBack,
                        _ => unreachable!(), // Handle unknown nullary operations
                    };
                    Some(GTStatement::Nullary(nullary_op))
                }
                old::Rule::unary_statement => {
                    let mut pairs = pair.into_inner();
                    let unary_op = match pairs.next().unwrap().as_str() {
                        "Click" => UnaryOp::Click,
                        "ScrapeText" => UnaryOp::ScrapeText,
                        "Download" => UnaryOp::Download,
                        "ScrapeLink" => UnaryOp::ScrapeLink,
                        _ => unreachable!(), // Handle unknown unary operations
                    };
                    Some(GTStatement::Unary(unary_op))
                }
                old::Rule::send_statement => {
                    let mut pairs = pair.into_inner();
                    let operator = match pairs.next().unwrap().as_str() {
                        "SendKeys" => GTStatement::SendKeys,
                        "SendData" => GTStatement::SendData,
                        _ => unreachable!(), // Handle unknown operator gracefully
                    };
                    Some(operator)
                }
                old::Rule::while_statement => {
                    let pairs = pair.into_inner();
                    let block = GTProgram::from(pairs);
                    Some(GTStatement::While(block))
                }
                old::Rule::for_statement => {
                    let pairs = pair.into_inner();
                    let block = GTProgram::from(pairs);
                    Some(GTStatement::For(block))
                }
                old::Rule::EOI => None,
                old::Rule::if_statement => None, // Skip the condition of While for now
                _ => panic!("Reached unhandled rule {:?}", pair.as_rule()), // Handle unknown rule gracefully
            })
            .collect();

        GTProgram::P(statements)
    }
}

impl From<Pairs<'_, new::Rule>> for GTProgram {
    fn from(value: Pairs<new::Rule>) -> Self {
        let statements = value
            .filter_map(|pair| match pair.as_rule() {
                new::Rule::nullary_statement => {
                    let nullary_op = match pair.as_str() {
                        "ExtractURL" => NullaryOp::ExtractURL,
                        "GoBack" => NullaryOp::GoBack,
                        _ => unreachable!(), // Handle unknown nullary operations
                    };
                    Some(GTStatement::Nullary(nullary_op))
                }
                new::Rule::unary_statement => {
                    let mut pairs = pair.into_inner();
                    let unary_op = match pairs.next().unwrap().as_str() {
                        "Click" => UnaryOp::Click,
                        "ScrapeText" => UnaryOp::ScrapeText,
                        "Download" => UnaryOp::Download,
                        "ScrapeLink" => UnaryOp::ScrapeLink,
                        _ => unreachable!(), // Handle unknown unary operations
                    };
                    Some(GTStatement::Unary(unary_op))
                }
                new::Rule::send_statement => {
                    let mut pairs = pair.into_inner();
                    let operator = match pairs.next().unwrap().as_str() {
                        "SendKeys" => GTStatement::SendKeys,
                        "SendData" => GTStatement::SendData,
                        _ => unreachable!(), // Handle unknown operator gracefully
                    };
                    Some(operator)
                }
                new::Rule::while_statement => {
                    let pairs = pair.into_inner();
                    let block = GTProgram::from(pairs);
                    Some(GTStatement::While(block))
                }
                new::Rule::for_statement => {
                    let pairs = pair.into_inner();
                    let block = GTProgram::from(pairs);
                    Some(GTStatement::For(block))
                }
                new::Rule::EOI => None,
                _ => panic!("Unhandled rule {:?}", pair.as_rule()), // Handle unknown statements
            })
            .collect();
        GTProgram::P(statements)
    }
}

pub fn parse_ground_truth(filename: &str) -> GTProgram {
    // Open the input file
    let content = std::fs::read_to_string(filename).expect("Failed to open file");

    if content.is_empty() {
        GTProgram::NGT
    } else if content.starts_with("NDSL") {
        GTProgram::NDSL
    } else {
        let parse_result = old::GTParser::parse(old::Rule::program, &content);
        parse_result
            .ok()
            .map(GTProgram::from)
            .or_else(|| {
                let new_result = new::GTParser::parse(new::Rule::program, &content);
                new_result.ok().map(GTProgram::from)
            })
            .unwrap()
    }
}

pub fn check_program(ground_truth: &GTProgram, program: &GTProgram) -> bool {
    println!("{:?}", ground_truth);
    println!("{:?}", program);
    ground_truth == program
}
