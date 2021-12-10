use crate::LinearSolver;
use na::DVector;
use nalgebra as na;
use nalgebra_sparse as nas;
use nas::factorization::CscCholesky;
use nas::CscMatrix;

pub struct CscCholeskySolver {}

impl LinearSolver for CscCholeskySolver {
    type MatrixType = CscMatrix<f64>;

    fn new() -> Self {
        CscCholeskySolver {}
    }

    #[allow(non_snake_case)]
    #[inline]
    fn solve(&self, A: &Self::MatrixType, b: &DVector<f64>) -> DVector<f64> {
        DVector::from_columns(&[CscCholesky::factor(A).unwrap().solve(b).column(0)])
    }
}
