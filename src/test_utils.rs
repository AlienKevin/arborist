use crate::{
    action::Actions,
    dom::DomTrace,
    dom_query,
    dsl::{program_to_string, NullaryOp, Program, UnaryOp},
    fta::Fta,
    predict::check_prediction,
    program_checker::{check_program, get_program_stats, parse_ground_truth, GTProgram},
    selector::Index,
    selector::{ConcreteSelector, SelectorPath},
    selector_matrix::Matrix,
    state_factory::{self, clean_up_states, clear_state_factory, get_state_factory_size},
    synthesis::synthesis,
    utils::value_to_object,
    value_path::ValuePath,
};

use super::{
    action::Action,
    selector_matrix::{ConcreteSelectorMatrix, Matrices},
    state::State,
    transition::Transitions,
};
use core::num;
use itertools::Itertools;
use regex::Regex;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use smallvec::smallvec;
use std::time::Duration;
use std::{fmt::Display, fs::OpenOptions, path::Path, str::FromStr, time::Instant};
use std::{
    fs::{self, File},
    io::Write,
};

#[derive(PartialEq, Eq, Clone, Copy)]
pub enum SliceTrace {
    True,
    False,
}

#[derive(PartialEq, Eq, Clone, Copy)]
pub enum PrintTrace {
    Print,
    Omit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BenchmarkId {
    pub w: u32,
    pub t: u32,
}

impl Serialize for BenchmarkId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl BenchmarkId {
    pub fn new(w: u32, t: u32) -> Self {
        Self { w, t }
    }
}

impl<'de> Deserialize<'de> for BenchmarkId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let input: String = serde::Deserialize::deserialize(deserializer)?;

        // Create a regex pattern to match the desired format
        let pattern = r"W(\d+)T(\d+)";
        let regex = Regex::new(pattern).unwrap();

        // Try to find a match in the input string
        if let Some(captures) = regex.captures(&input) {
            // Extract the captured number as a string
            if let (Some(w_str), Some(t_str)) = (captures.get(1), captures.get(2)) {
                // Parse the number string into an unsigned 32-bit integer
                if let (Ok(w), Ok(t)) =
                    (w_str.as_str().parse::<u32>(), t_str.as_str().parse::<u32>())
                {
                    return Ok(BenchmarkId { w, t });
                }
            }
        }
        Err(serde::de::Error::custom("Invalid BenchmarkId format"))
    }
}

impl std::fmt::Display for BenchmarkId {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "W{}T{}", self.w, self.t)
    }
}

#[derive(Debug, Serialize)]
pub struct ExperimentResult {
    id: BenchmarkId,
    length: u32,
    correct: bool,
    time: f64,
    incremental_cost: f64,
    synthesis_cost: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SummaryResult {
    pub name: BenchmarkId,
    pub length: usize,
    pub mean: f64,
    pub max: f64,
    pub acc: f64,
    pub in_pldi: String,
    pub intend: Intended,
    pub wr_solved: String,
    pub timeout: Option<usize>, // Timed out at which iteration
    pub sample_rate: f32,
    pub n_states: usize,
    pub n_transitions: usize,
    pub loop_depth: usize,
    pub loop_count: usize,
    pub ast_size: usize,
    pub mean_speculation_cost: f64,
    pub mean_validation_cost: f64,
    pub max_speculation_cost: f64,
    pub max_validation_cost: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ScaleExperimentResult {
    pub name: BenchmarkId,
    pub index: u32,
    pub length: usize,
    pub time_limit: u64,
    pub n_selectors: usize,
    pub time_cost: f64,
    pub timeout: String,
}

pub struct TimeStats {
    pub speculation_cost: f64,
    pub validation_cost: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct IntendedTraceLength {
    pub id: u32,
    pub length: usize,
}

pub fn check_states(result_states: &FxHashSet<State>, expected_states: &FxHashSet<State>) {
    assert_eq!(result_states.len(), expected_states.len());
    let mut expected_iter = expected_states.iter();
    for result in result_states {
        let expected = expected_iter.next().unwrap();
        assert!(result == expected);
    }
}

pub fn print_transitions(transitions: &Transitions) {
    for key in transitions.0.keys().sorted().rev() {
        println!("{:#?}: {:?}", key, transitions.0[key]);
    }
}

// Selects: body/div[@class=chosen][{div_index}]/button[@class=clicked][{button_index}]
pub fn select_button(div_index: usize, button_index: usize) -> ConcreteSelector {
    _select_button(true, "chosen", div_index, button_index)
}

// Selects: /div[@class=chosen][{div_index}]/button[@class=clicked][{button_index}]
pub fn select_button_short(div_index: usize, button_index: usize) -> ConcreteSelector {
    _select_button(false, "chosen", div_index, button_index)
}

// Creates an invalid selector on three_buttons.html
// body/div[@class=invalid][{div_index}]/button[@class=clicked][{button_index}]
pub fn select_button_invalid(div_index: usize, button_index: usize) -> ConcreteSelector {
    _select_button(true, "invalid", div_index, button_index)
}

// Selects: /div[@class={div_class}][{div_index}]/button[@class=clicked][{button_index}]
fn _select_button(
    body_as_root: bool,
    div_class: &'static str,
    div_index: usize,
    button_index: usize,
) -> ConcreteSelector {
    ConcreteSelector::from_str(&format!(
        "{body}/div[@class='{div_class}'][{div_index}]/button[@class='clicked'][{button_index}]",
        body = if body_as_root { "//body" } else { "" },
        div_class = div_class,
        div_index = div_index,
        button_index = button_index,
    ))
    .unwrap()
}

pub fn check_transitions_size(expected_transitions_size: usize, transitions: Transitions) {
    (0..expected_transitions_size).for_each(|state_id| {
        assert!(transitions.get_by_id(state_id).is_some());
    });
    assert_eq!(transitions.0.len(), expected_transitions_size);
}

use serde_json::Value;

pub type TraceActions = Vec<TraceAction>;

// const TIMEOUT: f64 = 1.0;

#[derive(Debug, Clone)]
pub enum TraceAction {
    Nullary(NullaryOp),
    Unary(UnaryOp, ConcreteSelectorMatrix),
    SendKeys(ConcreteSelectorMatrix, String),
    SendData(ConcreteSelectorMatrix, ValuePath), // TODO: add the rest of actions from WebRobot paper
}

impl TraceAction {
    pub fn from_json(json: Value) -> Option<Self> {
        let mut json = value_to_object(json).unwrap();
        let action_name = json["actionName"].as_str().unwrap();
        match action_name {
            "Click" | "ScrapeText" | "Download" | "ScrapeLink" => Some(TraceAction::Unary(
                UnaryOp::from_str(action_name).unwrap(),
                ConcreteSelectorMatrix::from_json(json.remove("targetElement").unwrap()),
            )),
            "GoBack" | "ExtractURL" => Some(TraceAction::Nullary(
                NullaryOp::from_str(action_name).unwrap(),
            )),
            "SendKeys" => Some(TraceAction::SendKeys(
                ConcreteSelectorMatrix::from_json(json.remove("targetElement").unwrap()),
                json.remove("data").unwrap().as_str().unwrap().to_string(),
            )),
            "SendData" => Some(TraceAction::SendData(
                ConcreteSelectorMatrix::from_json(json.remove("targetElement").unwrap()),
                ValuePath::from_str(json["data"].as_str().unwrap()).unwrap(),
            )),
            // TODO: Add the rest of actions
            _ => None,
        }
    }

    pub fn to_action(&self, action_index: Index) -> Action {
        match self {
            TraceAction::Nullary(op) => Action::Nullary(*op),
            TraceAction::Unary(op, _) => Action::Unary(action_index, *op),
            TraceAction::SendKeys(_, data) => Action::SendKeys(action_index, data.clone()),
            TraceAction::SendData(_, v) => Action::SendData(action_index, v.clone()),
        }
    }
}

pub fn load_trace(
    benchmark_path: &str,
    length: usize,
    slice: SliceTrace,
    print_trace: PrintTrace,
) -> TraceActions {
    let trace_path = benchmark_path.to_string() + "/trace.json";
    let trace: Value = serde_json::from_str(
        &fs::read_to_string(&trace_path)
            .unwrap_or_else(|_| panic!("Unable to read {}", trace_path)),
    )
    .unwrap();
    let trace = trace.as_object().unwrap();
    let actions = trace["actions"].as_array().unwrap();
    // TODO: remove the following line used for debugging
    let actions = if slice == SliceTrace::True {
        match benchmark_path {
            "tests/benchmarks/W228T1" => {
                [&actions[..7], &actions[67..71], &actions[113..]].concat()
            }
            "tests/benchmarks/W77T1" => [
                &actions[1..6],
                &actions[101..107],
                &actions[202..208],
                &actions[303..],
            ]
            .concat(),
            "tests/benchmarks/W162T1" => {
                [&actions[..8], &actions[40..47], &actions[105..]].concat()
            }
            "tests/benchmarks/W164T1" => [&actions[..9], &actions[27..35], &actions[54..]].concat(),
            _ => actions.to_vec(),
        }
    } else {
        actions.to_vec()
    };

    let mut filtered_trace_actions: Vec<TraceAction> = actions
        .into_iter()
        .filter_map(TraceAction::from_json)
        .take(length)
        .collect();
    if length == usize::MAX {
        // pop the last action in case there's not a dom afterwards
        filtered_trace_actions.pop();
    }
    if print_trace == PrintTrace::Print {
        for (action_index, action) in filtered_trace_actions.iter().enumerate() {
            println!("{}", action.to_action(action_index as Index));
        }
    }

    filtered_trace_actions
}

pub fn output_dot_file(fta: &Fta, name: &str) {
    let mut f = File::create(format!("{}.dot", name)).unwrap();
    writeln!(f, "{}", fta).expect("Unable to output dot file");
}

#[derive(Clone)]
pub struct BenchmarkConfig {
    pub max_selector_depth: usize,
    pub sample_rate: f32,
    pub max_big_n_start_index: Index,
}

pub fn round_to_decimal_places(number: f64, decimal_places: u32) -> f64 {
    let factor = 10.0_f64.powi(decimal_places as i32);
    (number * factor).round() / factor
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Intended {
    #[serde(rename = "Y")]
    Intended,
    #[serde(rename = "N")]
    Unintended,
    #[serde(rename = "NGT")]
    NoGroundTruth,
    #[serde(rename = "NDSL")] // not supported by current DSL
    NDSL,
}

impl std::fmt::Display for Intended {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", serde_plain::to_string(self).unwrap())
    }
}

const TIMEOUT_DURATION: Duration = Duration::from_millis(900);

struct IterationResult {
    program: Option<Program>,
    timeout: bool,
    experiment_result: ExperimentResult,
}

fn test_iteration(
    benchmark_id: BenchmarkId,
    name: &str,
    curr_length: usize,
    start_time: Instant,
    time_stats: &mut TimeStats,
    fta: &mut Fta,
    total_concrete_matrices: &Vec<Option<ConcreteSelectorMatrix>>,
    total_matrices: &Matrices,
    total_actions: &Actions,
    data: &Option<Value>,
    run_prediction: bool,
    config: &BenchmarkConfig,
    time_limit: Option<Duration>,
    num_selectors: Option<usize>,
) -> IterationResult {
    let eval_dom_trace = DomTrace::new(0, (curr_length + 1) as Index);

    let incremental_start_time = Instant::now();
    let mut result_program = None;

    let mut can_predict = false;
    if run_prediction {
        println!();
        println!("\n{name} start prediction {}", curr_length);

        let (program, _) = fta.extract_program(fta.root);
        println!("program: {}\n", program_to_string(&program, ""));

        if let Some(matrix) = &total_concrete_matrices[curr_length] {
            let ss = matrix.get_selectors(
                config.max_selector_depth,
                config.sample_rate,
                curr_length as u64,
            );
            println!("number of selectors: {}", ss.len());
        } else {
            println!("number of selectors: 0");
        }

        let predicted_action = crate::predict::run_program(
            &program,
            0,
            (curr_length + 1) as Index,
            &None,
            &Some(ValuePath::new()),
            total_matrices,
            data,
            total_actions,
        );
        if let Some(predicted_action) = &predicted_action {
            if check_prediction(
                &total_actions[curr_length],
                predicted_action,
                total_matrices,
            ) {
                can_predict = true;
            }
        }

        if can_predict {
            println!("Succeeded in prediction");
        } else {
            println!(
                "Failed in prediction.
Expected: {}
Found:    {}",
                &total_actions[curr_length],
                predicted_action.map_or("Empty".to_string(), |p| p.to_string()),
            );
        }
        result_program = Some(program);
    }
    let timeout_duration = if time_limit.is_some() {
        time_limit.unwrap()
    } else {
        TIMEOUT_DURATION
    };

    if start_time.elapsed() < timeout_duration {
        fta.add_next_action(
            total_actions[curr_length].clone(),
            &total_concrete_matrices[curr_length]
                .clone()
                .unwrap_or_else(|| ConcreteSelectorMatrix {
                    size: 0,
                    data: vec![],
                }),
            data,
            total_actions,
            total_matrices,
            eval_dom_trace,
            config,
            num_selectors,
            timeout_duration - start_time.elapsed(),
        );
    }

    let incremental_cost = incremental_start_time.elapsed().as_secs() as f64
        + incremental_start_time.elapsed().subsec_nanos() as f64 / 1e9;
    println!("incremental_cost: {}", incremental_cost);

    let synthesis_start_time = Instant::now();
    if start_time.elapsed() < timeout_duration {
        synthesis(
            fta,
            total_actions,
            total_matrices,
            data,
            config,
            timeout_duration - start_time.elapsed(),
            time_stats,
        );
    }

    // update time stats for the current iteration:
    time_stats.validation_cost += incremental_cost;

    let synthesis_cost = synthesis_start_time.elapsed().as_secs() as f64
        + synthesis_start_time.elapsed().subsec_nanos() as f64 / 1e9;

    println!("synthesis_cost: {}", synthesis_cost);
    clean_up_states(fta.bfs().collect());
    fta.clean_up(fta.root);

    let iteration_time =
        start_time.elapsed().as_secs() as f64 + start_time.elapsed().subsec_nanos() as f64 / 1e9;

    let experiment_result = ExperimentResult {
        id: benchmark_id,
        length: curr_length as u32,
        correct: can_predict,
        time: iteration_time,
        incremental_cost,
        synthesis_cost,
    };
    let timeout = start_time.elapsed() >= timeout_duration;
    IterationResult {
        program: result_program,
        timeout,
        experiment_result,
    }
}

pub fn test_experiment(
    name: &str,
    end_length: usize,
    result_path: &str,
    summary_path: &str,
    config: BenchmarkConfig,
    write_header: bool,
    is_incremental: bool,
) {
    crate::dom_query::setup_benchmark(name);
    let benchmark_id = serde_plain::from_str::<BenchmarkId>(name).unwrap();
    println!("=====evaluation result of {}=====", name);
    clear_state_factory();
    let mut correct_pred = 0;
    let total_trace_actions = load_trace(
        &format!("{}", name),
        end_length,
        SliceTrace::False,
        PrintTrace::Omit,
    );
    let trace_length = total_trace_actions.len();
    let total_concrete_matrices: Vec<Option<ConcreteSelectorMatrix>> = total_trace_actions
        .iter()
        .map(|trace_action| match trace_action {
            TraceAction::Nullary(_) => None,
            TraceAction::Unary(_, matrix) => Some(matrix.clone()),
            TraceAction::SendKeys(matrix, _) => Some(matrix.clone()),
            TraceAction::SendData(matrix, _) => Some(matrix.clone()),
        })
        .collect();

    let total_matrices: Vec<_> = total_trace_actions
        .iter()
        .enumerate()
        .map(|(index, trace_action)| match trace_action {
            TraceAction::Nullary(_) => FxHashMap::default(),
            TraceAction::Unary(_, matrix)
            | TraceAction::SendKeys(matrix, _)
            | TraceAction::SendData(matrix, _) => {
                let ns = matrix.get_selectors(config.max_selector_depth, 1.0, index as u64);
                let mut m: Matrix = FxHashMap::default();
                for SelectorPath { path, indices } in ns {
                    m.entry(path)
                        .and_modify(|i| i.push(indices.clone()))
                        .or_insert(smallvec![indices]);
                }
                m
            }
        })
        .collect();
    let file = OpenOptions::new()
        .append(true)
        .open(result_path)
        .expect("cannot open file");
    let mut csv_writer = csv::Writer::from_writer(file);

    let mut fta = Fta::default();

    // read input
    let input_path = format!("{}/input.json", name);
    let data: Option<Value> = if Path::new(&input_path).exists() {
        Some(
            serde_json::from_str(
                &fs::read_to_string(&input_path)
                    .unwrap_or_else(|_| panic!("Unable to read {}", input_path)),
            )
            .unwrap(),
        )
    } else {
        None
    };

    let exp_start_time = Instant::now();
    let mut max_time_cost: f64 = 0.0;
    let mut total_validation_cost: f64 = 0.0;
    let mut total_speculation_cost: f64 = 0.0;
    let mut max_validation_cost: f64 = 0.0;
    let mut max_speculation_cost: f64 = 0.0;

    let total_actions: Actions = total_trace_actions
        .iter()
        .enumerate()
        .map(|(action_index, trace_action)| trace_action.to_action(action_index as Index))
        .collect();

    let mut previous_program = vec![];
    let mut curr_length = 0;
    let mut has_timeout = false;
    while curr_length < trace_length {
        let start_time: Instant = Instant::now();

        dom_query::load_doc(curr_length as Index + 1);

        let mut iteration_result = None;
        let mut time_stats = TimeStats {
            speculation_cost: 0.0,
            validation_cost: 0.0,
        };
        if is_incremental {
            iteration_result = Some(test_iteration(
                benchmark_id,
                name,
                curr_length,
                start_time,
                &mut time_stats,
                &mut fta,
                &total_concrete_matrices,
                &total_matrices,
                &total_actions,
                &data,
                true,
                &config,
                None,
                None,
            ));
        } else {
            clear_state_factory();
            fta = Fta::default();
            for i in 0..=curr_length {
                iteration_result = Some(test_iteration(
                    benchmark_id,
                    name,
                    i,
                    start_time,
                    &mut time_stats,
                    &mut fta,
                    &total_concrete_matrices,
                    &total_matrices,
                    &total_actions,
                    &data,
                    i == curr_length,
                    &config,
                    None,
                    None,
                ));
            }
        }
        let IterationResult {
            program,
            timeout,
            experiment_result,
        } = iteration_result.unwrap();
        if experiment_result.correct {
            correct_pred += 1;
        }
        max_time_cost = max_time_cost.max(experiment_result.time);
        total_validation_cost += time_stats.validation_cost;
        total_speculation_cost += time_stats.speculation_cost;
        max_validation_cost = max_validation_cost.max(time_stats.validation_cost);
        max_speculation_cost = max_speculation_cost.max(time_stats.speculation_cost);
        csv_writer
            .serialize(experiment_result)
            .expect("Cannot serialize experiment result");
        csv_writer
            .flush()
            .expect("Cannot write experiment result to file");

        previous_program = program.unwrap();
        println!("{name} end prediction {}\n", curr_length);

        has_timeout = timeout;
        if timeout {
            break;
        }
        curr_length += 1;
    }

    let timeout = if has_timeout {
        println!(
            "Timed out at {} for benchmark {}",
            curr_length, benchmark_id
        );
        Some(curr_length)
    } else {
        println!(
            "final program: {}\n",
            program_to_string(&previous_program, "")
        );
        None
    };

    let (intended, loop_depth, loop_count, ast_size) = examine_program(name, previous_program);
    println!("Intended: {:?}", intended);
    let summary_result = SummaryResult {
        name: benchmark_id,
        length: trace_length,
        mean: exp_start_time.elapsed().as_secs() as f64 / trace_length as f64,
        max: max_time_cost,
        acc: correct_pred as f64 / (timeout.unwrap_or(trace_length) - 1) as f64,
        in_pldi: if PLDI_BENCHMARKS.contains(&benchmark_id) {
            "Y".to_string()
        } else {
            "N".to_string()
        },
        intend: intended,
        wr_solved: if WR_SOLVABLE_BENCHMARKS.contains(&benchmark_id) {
            "Y".to_string()
        } else {
            "N".to_string()
        },
        timeout,
        sample_rate: config.sample_rate,
        n_states: get_state_factory_size(),
        n_transitions: fta.transitions.size(),
        loop_depth,
        loop_count,
        ast_size,
        mean_speculation_cost: round_to_decimal_places(
            total_speculation_cost / trace_length as f64,
            6,
        ),
        mean_validation_cost: round_to_decimal_places(
            total_validation_cost / trace_length as f64,
            6,
        ),
        max_speculation_cost: round_to_decimal_places(max_speculation_cost, 6),
        max_validation_cost: round_to_decimal_places(max_validation_cost, 6),
    };
    // record summary result
    let file = OpenOptions::new()
        .append(true)
        .open(summary_path)
        .expect("cannot open file");
    let mut builder = csv::WriterBuilder::new();
    builder.has_headers(write_header);
    let mut csv_writer = builder.from_writer(file);
    // Check if the file is empty, write headers if necessary
    csv_writer
        .serialize(summary_result)
        .expect("Cannot serialize experiment result");
    csv_writer
        .flush()
        .expect("Cannot write experiment result to file");
}

pub fn test_scale_experiment(
    name: &str,
    index: u32,
    _start_length: usize,
    end_length: usize,
    result_path: &str,
    config: BenchmarkConfig,
    number_of_selectors: usize,
    write_header: bool,
    time_limit: u64,
) -> bool {
    crate::dom_query::setup_benchmark(name);
    let benchmark_id = serde_plain::from_str::<BenchmarkId>(name).unwrap();
    println!("=====evaluation result of {}=====", name);
    clear_state_factory();
    let total_trace_actions = load_trace(
        &format!("{}", name),
        end_length,
        SliceTrace::False,
        PrintTrace::Omit,
    );
    let trace_length = total_trace_actions.len();
    let total_concrete_matrices: Vec<Option<ConcreteSelectorMatrix>> = total_trace_actions
        .iter()
        .map(|trace_action| match trace_action {
            TraceAction::Nullary(_) => None,
            TraceAction::Unary(_, matrix) => Some(matrix.clone()),
            TraceAction::SendKeys(matrix, _) => Some(matrix.clone()),
            TraceAction::SendData(matrix, _) => Some(matrix.clone()),
        })
        .collect();

    let total_matrices: Matrices = total_trace_actions
        .iter()
        .map(|trace_action| match trace_action {
            TraceAction::Nullary(_) => FxHashMap::default(),
            TraceAction::Unary(_, matrix)
            | TraceAction::SendKeys(matrix, _)
            | TraceAction::SendData(matrix, _) => {
                let ns = matrix.get_num_selectors(number_of_selectors);
                let mut m: Matrix = FxHashMap::default();
                for SelectorPath { path, indices } in ns {
                    m.entry(path)
                        .and_modify(|i| i.push(indices.clone()))
                        .or_insert(smallvec![indices]);
                }
                m
            }
        })
        .collect();

    // read input
    let input_path = format!("{}/input.json", name);
    let data: Option<Value> = if Path::new(&input_path).exists() {
        Some(
            serde_json::from_str(
                &fs::read_to_string(&input_path)
                    .unwrap_or_else(|_| panic!("Unable to read {}", input_path)),
            )
            .unwrap(),
        )
    } else {
        None
    };

    // parse file name
    let total_actions: Actions = total_trace_actions
        .iter()
        .enumerate()
        .map(|(action_index, trace_action)| trace_action.to_action(action_index as Index))
        .collect();

    let mut max_time_cost: f64 = 0.0;
    let curr_length = trace_length - 1;

    println!();
    println!("\n{name} start prediction {}", curr_length);

    // let (_program, _) = fta.extract_program(fta.root);
    // incrementally synthesize the fta
    let mut fta = Fta::default();
    // output_dot_file(&fta, &format!("scale_{}", curr_length));

    let start_time: Instant = Instant::now();
    let mut iteration_result = None;
    let mut time_stats = TimeStats {
        speculation_cost: 0.0,
        validation_cost: 0.0,
    };
    for i in 0..=curr_length {
        iteration_result = Some(test_iteration(
            benchmark_id,
            name,
            i,
            start_time,
            &mut time_stats,
            &mut fta,
            &total_concrete_matrices,
            &total_matrices,
            &total_actions,
            &data,
            i == curr_length,
            &config,
            Some(Duration::from_millis(time_limit)),
            Some(number_of_selectors),
        ));
    }
    let IterationResult {
        program,
        timeout,
        experiment_result,
    } = iteration_result.unwrap();

    let exp_end_time = Instant::now();
    let exp_time = exp_end_time.duration_since(start_time);

    // max_time_cost = max_time_cost.max(experiment_result.time);
    let timeout = exp_time > Duration::from_millis(time_limit);
    println!("{name} end prediction {}\n", curr_length);

    let experiment_result = ScaleExperimentResult {
        name: benchmark_id,
        index,
        length: end_length,
        time_limit,
        n_selectors: number_of_selectors,
        time_cost: start_time.elapsed().as_secs() as f64
            + start_time.elapsed().subsec_nanos() as f64 / 1e9,
        timeout: if timeout {
            "Y".to_string()
        } else {
            "N".to_string()
        },
    };
    let final_program = fta.extract_program(fta.root).0;
    println!("final program:\n{}", program_to_string(&final_program, ""));
    let file = OpenOptions::new()
        .append(true)
        .open(result_path)
        .expect("cannot open file");
    let mut builder = csv::WriterBuilder::new();
    builder.has_headers(write_header);
    let mut csv_writer = builder.from_writer(file);
    // Check if the file is empty, write headers if necessary
    csv_writer
        .serialize(experiment_result)
        .expect("Cannot serialize experiment result");
    csv_writer
        .flush()
        .expect("Cannot write experiment result to file");
    timeout
}

fn examine_program(name: &str, previous_program: Program) -> (Intended, usize, usize, usize) {
    let mut loop_depth = 0;
    let mut loop_count = 0;
    let mut ast_size = 0;
    let folder_path = Path::new(name);
    let intended = if let Some(name) = find_program_file(folder_path) {
        println!("Found program file: {}", name);
        let program_path = folder_path.join(name);
        let program_filename = program_path.to_str().unwrap();
        let ground_truth = parse_ground_truth(program_filename);
        (loop_depth, loop_count, ast_size) = get_program_stats(&ground_truth);
        match ground_truth {
            GTProgram::NGT => Intended::NoGroundTruth,
            GTProgram::NDSL => Intended::NDSL,
            GTProgram::P(_) => {
                let previous_program = GTProgram::from(previous_program);
                if check_program(&ground_truth, &previous_program) {
                    Intended::Intended
                } else {
                    Intended::Unintended
                }
            }
        }
    } else {
        println!("Program file not found.");
        Intended::NoGroundTruth
    };
    (intended, loop_depth, loop_count, ast_size)
}

pub fn find_program_file(path: &Path) -> Option<String> {
    for entry in fs::read_dir(path).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_dir() {
            if let Some(name) = find_program_file(&path) {
                return Some(name);
            }
        } else if let Some(ext) = path.extension() {
            if ext == "program" {
                if let Some(name) = path.file_name() {
                    return Some(name.to_string_lossy().to_string());
                }
            }
        }
    }
    None
}

// Points must be sorted in ascending order and start at 1
pub struct InflectionPoints {
    points: Vec<usize>,
}

impl InflectionPoints {
    pub fn new(points: Vec<usize>) -> Self {
        Self { points }
    }

    pub fn can_predict(&self, point_in_question: usize) -> bool {
        use itertools::FoldWhile::*;
        let last_can_predict = self.points.len() % 2 != 0;
        // Source: https://www.reddit.com/r/rust/comments/eq4l8s/comment/ferpxtu/?utm_source=share&utm_medium=web2x&context=3
        self.points
            .iter()
            .rev()
            .fold_while(last_can_predict, |can_predict, &point| {
                if point_in_question >= point {
                    Done(can_predict)
                } else {
                    Continue(!can_predict)
                }
            })
            .into_inner()
    }
}

#[derive(Debug, Serialize, Clone)]
pub struct VerifyResult {
    pub name: String,
    pub length: u32,
    pub correct: bool,
    pub can_predict: bool,
    pub time: f32,
}

impl Display for VerifyResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}\t{}\t{}",
            self.length,
            self.can_predict,
            self.time.round()
        )
    }
}

pub enum VerifyCorrect {
    Correct,
    Wrong(Vec<VerifyResult>),
}

impl Display for VerifyCorrect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Correct => write!(f, "Verification Success"),
            Self::Wrong(results) => {
                writeln!(f, "length\tpred\ttime")?;
                for result in results {
                    writeln!(f, "{}", result)?;
                }
                Ok(())
            }
        }
    }
}

lazy_static::lazy_static! {
    static ref PLDI_BENCHMARKS: Vec<BenchmarkId> = vec![
        (1, 1),
        (3, 1),
        (6, 1),
        (8, 1),
        (9, 1),
        (14, 1),
        (18, 1),
        (25, 1),
        (33, 1),
        (34, 1),
        (38, 1),
        (40, 1),
        (46, 1),
        (49, 1),
        (50, 1),
        (52, 1),
        (58, 1),
        (69, 1),
        (77, 1),
        (81, 1),
        (87, 1),
        (88, 1),
        (111, 1),
        (115, 1),
        (125, 1),
        (127, 1),
        (133, 1),
        (134, 1),
        (138, 1),
        (141, 1),
        (144, 1),
        (146, 1),
        (148, 1),
        (149, 1),
        (157, 1),
        (162, 1),
        (177, 1),
        (178, 1),
        (188, 1),
        (190, 1),
        (204, 1),
        (213, 1),
        (223, 1),
        (226, 1),
        (228, 1),
        (232, 1),
        (233, 1),
        (237, 1),
        (238, 1),
        (239, 1),
        (240, 1),
        (252, 1),
        (253, 1),
        (254, 1),
        (262, 1),
        (265, 1),
        (268, 1),
        (274, 1),
        (276, 1),
        (284, 1),
        (285, 1),
        (287, 1),
        (296, 1), // former sz2 benchmarks
        (99, 1),
        (164, 1),
        (176, 1),
        (189, 1),
        (295, 1), // former err benchmarks
        (48, 1),
        (53, 1),
        (56, 1),
        (80, 1),
        (112, 1),
        (156, 1),
        (205, 1),
        (218, 1),
    ]
    .iter()
    .map(|(w, t)| BenchmarkId::new(*w, *t))
    .collect();

    static ref WR_SOLVABLE_BENCHMARKS: Vec<BenchmarkId> = vec![
        (1, 1),
        (3, 1),
        (6, 1),
        (8, 1),
        (9, 1),
        (14, 1),
        (18, 1),
        (25, 1),
        (33, 1),
        (34, 1),
        (38, 1),
        (40, 1),
        (46, 1),
        (49, 1),
        (50, 1),
        (52, 1),
        (58, 1),
        (77, 1),
        (81, 1),
        (87, 1),
        (88, 1),
        (111, 1),
        (115, 1),
        (125, 1),
        (127, 1),
        (133, 1),
        (134, 1),
        (138, 1),
        (141, 1),
        (144, 1),
        (146, 1),
        (148, 1),
        (149, 1),
        (157, 1),
        (162, 1),
        (177, 1),
        (178, 1),
        (188, 1),
        (190, 1),
        (204, 1),
        (213, 1),
        (223, 1),
        (226, 1),
        (228, 1),
        (232, 1),
        (233, 1),
        (237, 1),
        (238, 1),
        (239, 1),
        (240, 1),
        (252, 1),
        (253, 1),
        (254, 1),
        (262, 1),
        (265, 1),
        (268, 1),
        (274, 1),
        (276, 1),
        (284, 1),
        (285, 1),
        (287, 1),
        (296, 1), // former sz2 benchmarks
        (99, 1),
        (164, 1),
        (176, 1),
        (189, 1),
        (295, 1), // former err benchmarks
        (53, 1),
        (1, 2),
        (34, 2),
        (49, 2),
        (54, 2),
        (78, 2),
        (87, 2),
        (111, 2),
        (144, 2),
        (146, 2),
        (149, 2),
        (157, 2),
        (158, 2),
        (218, 2),
        (232, 2),
        (252, 2),
        (265, 2),
        (304, 1),
        (307, 1),
        (51, 2),
    ]
    .iter()
    .map(|(w, t)| BenchmarkId::new(*w, *t))
    .collect();
}
