use nalgebra::DVector;
use nalgebra_sparse::CsrMatrix;

use crate::LinearSolver;

pub struct MINRESLinear {}

impl LinearSolver for MINRESLinear {
    type MatrixType = CsrMatrix<f64>;
    fn new() -> Self {
        MINRESLinear {}
    }
    #[allow(non_snake_case)]
    fn solve(&self, _A: &Self::MatrixType, _b: &DVector<f64>) -> DVector<f64> {
        todo!()
    }
}
