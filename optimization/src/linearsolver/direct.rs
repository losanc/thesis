use crate::LinearSolver;
use na::{DMatrix, DVector};
use nalgebra as na;

pub struct DirectLinear {}

impl LinearSolver for DirectLinear {
    type MatrixType = DMatrix<f64>;
    fn new() -> Self {
        DirectLinear {}
    }
    #[allow(non_snake_case)]
    fn solve(&self, A: &Self::MatrixType, b: &DVector<f64>) -> DVector<f64> {
        let A = A.clone();
        A.try_inverse().unwrap() * b
    }
}

pub struct PivLU {}

impl LinearSolver for PivLU {
    type MatrixType = DMatrix<f64>;
    fn new() -> Self {
        PivLU {}
    }
    #[allow(non_snake_case)]
    fn solve(&self, A: &Self::MatrixType, b: &DVector<f64>) -> DVector<f64> {
        let dec = A.clone().full_piv_lu();
        let res = dec.solve(b).unwrap();
        res
    }
}
