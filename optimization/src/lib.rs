use na::{DMatrix, DVector};
use nalgebra as na;
use nalgebra_sparse as nas;
use nas::{CooMatrix, CscMatrix, CsrMatrix};

pub use crate::linearsolver::*;
pub use crate::linesearch::*;
pub use crate::solver::*;

pub trait MatrixType: std::fmt::Debug {
    fn mul(&self, v: &DVector<f64>) -> DVector<f64>;
    fn identity(&self) -> Self;
    fn inverse_diagoanl(&self) -> Self;
    fn to_dmatrix(&self) -> DMatrix<f64>;
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

    fn to_dmatrix(&self) -> DMatrix<f64> {
        self.clone()
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

    fn to_dmatrix(&self) -> DMatrix<f64> {
        DMatrix::<f64>::from(self)
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
    fn to_dmatrix(&self) -> DMatrix<f64> {
        DMatrix::<f64>::from(self)
    }
}

pub trait Problem {
    type HessianType: MatrixType;
    fn apply(&self, _x: &DVector<f64>) -> f64;
    fn gradient(&self, _x: &DVector<f64>) -> Option<DVector<f64>> {
        None
    }
    fn gradient_mut(&mut self, _x: &DVector<f64>) -> Option<DVector<f64>> {
        None
    }
    fn hessian(&self, _x: &DVector<f64>) -> Option<Self::HessianType> {
        None
    }
    fn hessian_mut(&mut self, _x: &DVector<f64>) -> Option<Self::HessianType> {
        None
    }

    fn hessian_inverse_mut(&mut self, _x: &DVector<f64>) -> Option<DMatrix<f64>> {
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
    fn search(&self, pro: &P, current: &DVector<f64>, direction: &DVector<f64>) -> f64;
    // fn check(&self, pro: &P, current: &DVector<f64>, direction: DVector<f64>) -> bool {
    //     true
    // }
}

pub trait Solver<P: Problem, L: LinearSolver<MatrixType = P::HessianType>, LS: LineSearch<P>>:
    Sized
{
    fn epi(&self) -> f64;
    fn solve<T: std::io::Write>(
        &self,
        pro: &mut P,
        lin: &L,
        ls: &LS,
        input: &DVector<f64>,
        log: &mut T,
    ) -> DVector<f64>;
}

#[macro_export]
macro_rules! mylog {
    ($log:expr, $infor:tt, $value:expr) => {
        #[cfg(feature = "log")]
        {
            writeln!($log, "{} : {}", $infor, $value).unwrap();
        }
    };
}

impl MatrixType for CsrMatrix<f64> {
    #[inline]
    fn mul(&self, v: &DVector<f64>) -> DVector<f64> {
        let (row_offsets, col_indices, values) = self.csr_data();
        let mut res = DVector::<f64>::zeros(v.len());
        let res_slice = res.as_mut_slice();
        let v_slice = v.as_slice();
        let mut acc0;
        let mut acc1;
        let mut acc2;
        let mut acc3;
        unsafe {
            for row in 0..row_offsets.len() - 1 {
                acc0 = 0.0;
                acc1 = 0.0;
                acc2 = 0.0;
                acc3 = 0.0;
                let start = *row_offsets.get_unchecked(row);
                let end = *row_offsets.get_unchecked(row + 1);
                let v_range = values.get_unchecked(start..end).chunks_exact(4);
                let c_range = col_indices.get_unchecked(start..end).chunks_exact(4);
                for (v, c_c) in v_range.zip(c_range) {
                    acc0 += v[0] * v_slice.get_unchecked(c_c[0]);
                    acc1 += v[1] * v_slice.get_unchecked(c_c[1]);
                    acc2 += v[2] * v_slice.get_unchecked(c_c[2]);
                    acc3 += v[3] * v_slice.get_unchecked(c_c[3]);
                }
                *res_slice.get_unchecked_mut(row) += (acc0 + acc1) + (acc2 + acc3);

                for col in ((end - start) / 4) * 4 + start..end {
                    *res_slice.get_unchecked_mut(row) += values.get_unchecked(col)
                        * v_slice.get_unchecked(*col_indices.get_unchecked(col));
                }
            }
        }
        res
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
    fn to_dmatrix(&self) -> DMatrix<f64> {
        DMatrix::<f64>::from(self)
    }
}
