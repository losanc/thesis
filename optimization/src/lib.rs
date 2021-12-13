use na::{DMatrix, DVector};
use nalgebra as na;
use nalgebra_sparse as nas;
use nas::{CooMatrix, CscMatrix, CsrMatrix};

pub trait MatrixType {
    fn mul(&self, v: &DVector<f64>) -> DVector<f64>;
    fn identity(&self) -> Self;
    fn inverse_diagoanl(&self) -> Self;
}
impl MatrixType for DMatrix<f64> {
    #[inline]
    fn mul(&self, v: &DVector<f64>) -> DVector<f64> {
        self * v
    }
    #[inline]
    fn identity(&self) -> Self {
        Self::identity(self.nrows(), self.ncols())
    }
    #[inline]
    fn inverse_diagoanl(&self) -> Self {
        DMatrix::<f64>::from_diagonal(&(self.map_diagonal(|x| 1.0 / x)))
    }
}

impl MatrixType for CsrMatrix<f64> {
    #[inline]
    fn mul(&self, v: &DVector<f64>) -> DVector<f64> {
        self * v
    }
    #[inline]
    fn identity(&self) -> Self {
        Self::identity(self.nrows())
    }
    #[inline]
    fn inverse_diagoanl(&self) -> Self {
        let mut mat = self.diagonal_as_csr();
        mat.values_mut().iter_mut().for_each(|x| *x = 1.0 / *x);
        mat
    }
}
impl MatrixType for CscMatrix<f64> {
    #[inline]
    fn mul(&self, v: &DVector<f64>) -> DVector<f64> {
        self * v
    }
    #[inline]
    fn identity(&self) -> Self {
        Self::identity(self.nrows())
    }
    #[inline]
    fn inverse_diagoanl(&self) -> Self {
        let mut mat = self.diagonal_as_csc();
        mat.values_mut().iter_mut().for_each(|x| *x = 1.0 / *x);
        mat
    }
}

impl MatrixType for CooMatrix<f64> {
    #[inline]
    fn mul(&self, v: &DVector<f64>) -> DVector<f64> {
        self * v
    }
    #[inline]
    fn identity(&self) -> Self {
        Self::identity(self.nrows())
    }
    #[inline]
    fn inverse_diagoanl(&self) -> Self {
        let mut mat = self.diagonal_as_coo();
        mat.compress();
        assert_eq!(mat.nnz(), mat.nrows());
        mat.values_mut().iter_mut().for_each(|x| *x = 1.0 / *x);
        mat
    }
}

pub trait Problem {
    type HessianType: MatrixType;
    fn apply(&self, _x: &DVector<f64>) -> f64;
    fn gradient(&self, _x: &DVector<f64>) -> Option<DVector<f64>> {
        None
    }
    fn hessian(&self, _x: &DVector<f64>) -> Option<Self::HessianType> {
        None
    }
}

pub mod linearsolver;
pub mod linesearch;
pub mod solver;

pub trait LinearSolver {
    type MatrixType: MatrixType;
    fn new() -> Self;
    #[allow(non_snake_case)]
    fn solve(&self, A: &Self::MatrixType, b: &DVector<f64>) -> DVector<f64>;
}

pub trait LineSearch<P: Problem> {
    fn search(&self, pro: &P, current: &DVector<f64>, direction: DVector<f64>) -> DVector<f64>;
}

pub trait Solver<P: Problem, L: LinearSolver<MatrixType = P::HessianType>, LS: LineSearch<P>>:
    Sized
{
    fn solve(&self, pro: &P, lin: &L, ls: &LS, input: &DVector<f64>) -> DVector<f64>;
}
