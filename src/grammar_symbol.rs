/// A grammar symbol used in `State`
///
/// Ensures no cycling is possible in our transitions
#[derive(Hash, Clone, Copy, PartialEq, Eq, Debug)]
pub enum GrammarSymbol {
    Program,
    Expr,
}
