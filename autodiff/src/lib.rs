mod gradient;
mod hessian;
pub use gradient::{
    constant_matrix_to_gradients, gradients_to_vector, vector_to_gradients, Gradient,
};
use std::ops::{Mul, Sub};

pub use hessian::{constant_matrix_to_hessians, hessians_to_vector, vector_to_hessians, Hessian};
use nalgebra::{ClosedAdd, ClosedMul, ClosedSub, SMatrix, SVector, Scalar};
use num::{One, Zero};

pub trait MyLog: Copy {
    fn myln(self) -> Self;
}

impl MyLog for f32 {
    #[inline]
    fn myln(self) -> Self {
        self.ln()
    }
}

impl MyLog for f64 {
    #[inline]
    fn myln(self) -> Self {
        self.ln()
    }
}

pub trait MyScalar:
    Scalar
    + Zero
    + One
    + ClosedAdd
    + ClosedMul
    + ClosedSub
    + Copy
    + Copy
    + MyLog
    + Mul<f64, Output = Self>
    + Sub<f64, Output = Self>
{
    fn as_myscalar_vec<const S: usize>(vec: SVector<f64, S>) -> SVector<Self, S>;
    fn as_constant_mat<const S: usize>(vec: SMatrix<f64, S, S>) -> SMatrix<Self, S, S>;
}
impl MyScalar for f64 {
    #[inline]
    fn as_myscalar_vec<const S: usize>(vec: SVector<f64, S>) -> SVector<Self, S> {
        vec
    }
    #[inline]
    fn as_constant_mat<const S: usize>(vec: SMatrix<f64, S, S>) -> SMatrix<Self, S, S> {
        vec
    }
}
