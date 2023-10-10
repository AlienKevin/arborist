use crate::io_pair::IOPair;

/// A set of annotations
///
/// Additional annotations may be added to form a new annotated
/// state during evaluation
pub type Annotations = Vec<IOPair>;
