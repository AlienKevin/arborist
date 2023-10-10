use crate::{
    selector::{ConcreteSelector, Index, Indices, SelectorPath, SelectorPathRaw, SelectorSegments},
    utils::value_to_array,
};
use internment::Intern;
use itertools::Itertools;
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use rustc_hash::FxHashMap;
use serde_json::Value;
use smallvec::{smallvec, SmallVec};

pub type Matrix = FxHashMap<SelectorSegments, SmallVec<[Indices; 1]>>;
pub type Matrices = Vec<Matrix>;

#[derive(Debug, Clone)]
pub struct ConcreteSelectorMatrix {
    pub size: usize,
    pub data: Vec<SelectorPathRaw>,
}

#[inline]
pub fn validate_selector(matrices: &Matrices, matrix_index: Index, rho: &ConcreteSelector) -> bool {
    if let ConcreteSelector::Path(path) = rho {
        if let Some(indices) = matrices[matrix_index as usize].get(&path.path) {
            indices.contains(&path.indices)
        } else {
            false
        }
    } else {
        panic!(
            "Calling validate_selector on something that's not a Path: {}",
            rho
        )
    }
}

impl ConcreteSelectorMatrix {
    pub fn to_absolute_selector(&self) -> SelectorPath {
        let mut segments = smallvec![];
        let mut indices = smallvec![];
        for parent_element_index in 0..self.size - 1 {
            let child_element_index = parent_element_index + 1;
            let parent_to_child_paths =
                self.get_selector_segments(parent_element_index, child_element_index);
            let (segment, index) = parent_to_child_paths
                .into_iter()
                .find(|p| p.0 .1.attr.is_none())
                .unwrap();
            segments.push(segment);
            indices.push(index);
        }
        SelectorPath {
            path: Intern::new(segments),
            indices,
        }
    }

    fn get_selector_segments(
        &self,
        ancestor_element_index: usize,
        descendant_element_index: usize,
    ) -> SelectorPathRaw {
        assert!(ancestor_element_index < descendant_element_index);

        self.data[ancestor_element_index * self.size + descendant_element_index].clone()
    }

    pub fn get_selectors(
        &self,
        max_selector_depth: usize,
        sample_rate: f32,
        index: u64,
    ) -> Vec<SelectorPath> {
        let mut result_selectors: Vec<SelectorPathRaw> = vec![];
        // depth = 0
        // the selector path from root to the target node
        self.get_selector_segments(0, self.size - 1)
            .into_iter()
            .for_each(|((rel, pred), i)| {
                result_selectors.push(SelectorPathRaw {
                    path: smallvec![(rel, pred)],
                    indices: smallvec![i],
                })
            });
        for depth in 1..(max_selector_depth + 1) {
            // A selector path must start at the root and end at the target element
            let index_combinations = (1..self.size - 1).combinations(depth);
            for combination in index_combinations {
                let mut selector_segments: Vec<SelectorPathRaw> = vec![];
                // Add the root as the start of the path
                selector_segments
                    .push(self.get_selector_segments(0, *combination.first().unwrap()));
                for (i, index) in combination.iter().enumerate() {
                    let next_index = if i == combination.len() - 1 {
                        // index of the target node
                        // The target node must be at the end of the path
                        self.size - 1
                    } else {
                        combination[i + 1]
                    };
                    let segment = self.get_selector_segments(*index, next_index);
                    selector_segments.push(segment);
                }
                let (paths, indices): (Vec<_>, Vec<_>) = selector_segments
                    .into_iter()
                    .map(|n| (n.path, n.indices))
                    .unzip();

                let result_paths = paths
                    .into_iter()
                    .multi_cartesian_product()
                    .map(SmallVec::from);
                let result_indices = indices
                    .into_iter()
                    .multi_cartesian_product()
                    .map(SmallVec::from);
                result_selectors.extend(
                    result_paths
                        .zip(result_indices)
                        .map(|(path, indices)| SelectorPathRaw { path, indices }),
                );
            }
        }
        let mut total_selectors: Vec<_> = result_selectors
            .into_iter()
            .map(|path| path.intern())
            .collect();

        // add absolute path selectors
        // for s in self.to_absolute_selector() {
        //     // total_selectors.push(s);
        //     println!("absolute selector: {}", s)
        // }
        let absolute_selector = self.to_absolute_selector();
        // println!("absolute selector: {}", absolute_selector);
        total_selectors.push(absolute_selector);

        let selector_count = total_selectors.len();
        let mut sample_count = (selector_count as f32 * sample_rate).ceil() as usize;
        if sample_count < 1 {
            // panic!(
            //     "Not enough selectors to sample: selector # {}, sample rate {}",
            //     selector_count, sample_rate
            // );
            sample_count = 1;
        }
        let mut rng = StdRng::seed_from_u64(index);
        total_selectors.shuffle(&mut rng);

        let selected_elements: Vec<_> = total_selectors
            .into_iter()
            .take(sample_count)
            .collect::<Vec<_>>();

        selected_elements
    }

    pub fn get_num_selectors(&self, num_of_selectors: usize) -> Vec<SelectorPath> {
        let mut result_selectors: Vec<SelectorPathRaw> = vec![];
        // depth = 0
        // the selector path from root to the target node
        if num_of_selectors == 1 {
            let absolute_selector = self.to_absolute_selector();
            return vec![absolute_selector];
        }

        for ((rel, pred), i) in self.get_selector_segments(0, self.size - 1).into_iter() {
            if result_selectors.len() >= num_of_selectors {
                return result_selectors
                    .into_iter()
                    .map(|path| path.intern())
                    .collect();
            }
            result_selectors.push(SelectorPathRaw {
                path: smallvec![(rel, pred)],
                indices: smallvec![i],
            })
        }

        for depth in 1..(self.size + 1) {
            // A selector path must start at the root and end at the target element
            if result_selectors.len() >= num_of_selectors {
                break;
            }
            let index_combinations = (1..self.size - 1).combinations(depth);
            for combination in index_combinations {
                let mut selector_segments: Vec<SelectorPathRaw> = vec![];
                // Add the root as the start of the path
                selector_segments
                    .push(self.get_selector_segments(0, *combination.first().unwrap()));
                for (i, index) in combination.iter().enumerate() {
                    let next_index = if i == combination.len() - 1 {
                        // index of the target node
                        // The target node must be at the end of the path
                        self.size - 1
                    } else {
                        combination[i + 1]
                    };
                    let segment = self.get_selector_segments(*index, next_index);
                    selector_segments.push(segment);
                }
                let (paths, indices): (Vec<_>, Vec<_>) = selector_segments
                    .into_iter()
                    .map(|n| (n.path, n.indices))
                    .unzip();

                let result_paths = paths
                    .into_iter()
                    .multi_cartesian_product()
                    .map(SmallVec::from);
                let result_indices = indices
                    .into_iter()
                    .multi_cartesian_product()
                    .map(SmallVec::from);
                let selectors_at_depth = result_paths
                    .zip(result_indices)
                    .map(|(path, indices)| SelectorPathRaw { path, indices })
                    .collect::<Vec<_>>();
                if selectors_at_depth.len() + result_selectors.len() > num_of_selectors {
                    let mut rng = StdRng::seed_from_u64(123);
                    let num_choices = num_of_selectors - result_selectors.len();
                    // Shuffle the vector using the random number generator
                    let mut shuffled = selectors_at_depth.clone();
                    shuffled.shuffle(&mut rng);

                    // Take the first `num_choices` items from the shuffled vector
                    let chosen = shuffled
                        .iter()
                        .take(num_choices)
                        .cloned()
                        .collect::<Vec<_>>();
                    result_selectors.extend(chosen);
                    break;
                }
                result_selectors.extend(selectors_at_depth);
            }
        }
        result_selectors
            .into_iter()
            .map(|path| path.intern())
            .collect()
    }

    pub fn from_json(json: Value) -> Self {
        let rows = value_to_array(json).unwrap();
        let size = rows.len();
        let mut data = vec![];
        for row in rows {
            let row = row.as_array().unwrap();
            for i in row {
                match i {
                    Value::Null => {
                        data.push(SelectorPathRaw::default());
                    }
                    Value::Array(alternative_selectors) => {
                        let (path, indices): (SmallVec<_>, SmallVec<_>) = alternative_selectors
                            .iter()
                            .map(|s| {
                                let mut n =
                                    ConcreteSelector::path_from_str(s.as_str().unwrap()).unwrap();
                                assert_eq!(n.len(), 1); // all selector segments in the matrix must have unit length
                                                        // get the first and only element from the list
                                (n.path.swap_remove(0), n.indices.swap_remove(0))
                            })
                            .unzip();
                        data.push(SelectorPathRaw { path, indices });
                    }
                    _ => {
                        panic!("Invalid concrete selector in matrix: {}", i)
                    }
                }
            }
        }
        ConcreteSelectorMatrix { size, data }
    }
}
