use na::{SMatrix, SVector};
use nalgebra as na;
use num::{One, Zero};
use std::fmt;
use std::ops::Div;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};
#[derive(Clone, PartialEq, Debug, Copy)]
pub struct Hessian<const N: usize> {
    value: f64,
    gradient: SVector<f64, N>,
    //  !!!IMPORTANT: only half of hessian
    hessian: SMatrix<f64, N, N>,
}

impl<const N: usize> Hessian<N> {
    pub fn hessian(&self) -> SMatrix<f64, N, N> {
        self.hessian + self.hessian.transpose()
    }
    pub fn gradient(&self) -> SVector<f64, N> {
        self.gradient
    }
    pub fn value(&self) -> f64 {
        self.value
    }
    pub fn as_constant(&mut self) {
        self.gradient = SVector::<f64, N>::zeros();
        self.hessian = SMatrix::<f64, N, N>::zeros();
    }
    pub fn to_constant(self) -> Self {
        Self {
            value: self.value,
            gradient: SVector::<f64, N>::zeros(),
            hessian: SMatrix::<f64, N, N>::zeros(),
        }
    }
}

impl<const N: usize> fmt::Display for Hessian<N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

// ----------------Operators begin------------------

impl<const N: usize> Add<Self> for Hessian<N> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Hessian {
            value: self.value + rhs.value,
            gradient: self.gradient + rhs.gradient,
            hessian: self.hessian + rhs.hessian,
        }
    }
}

impl<const N: usize> Add<f64> for Hessian<N> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: f64) -> Self {
        Hessian {
            value: self.value + rhs,
            gradient: self.gradient,
            hessian: self.hessian,
        }
    }
}

impl<const N: usize> AddAssign<Self> for Hessian<N> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.value += rhs.value;
        self.gradient += rhs.gradient;
        self.hessian += rhs.hessian;
    }
}

impl<const N: usize> AddAssign<f64> for Hessian<N> {
    #[inline]
    fn add_assign(&mut self, rhs: f64) {
        self.value += rhs;
    }
}

impl<const N: usize> Sub<Self> for Hessian<N> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Hessian {
            value: self.value - rhs.value,
            gradient: self.gradient - rhs.gradient,
            hessian: self.hessian - rhs.hessian,
        }
    }
}

impl<const N: usize> Sub<f64> for Hessian<N> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: f64) -> Self {
        Hessian {
            value: self.value - rhs,
            gradient: self.gradient,
            hessian: self.hessian,
        }
    }
}

impl<const N: usize> SubAssign<Self> for Hessian<N> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.value -= rhs.value;
        self.gradient -= rhs.gradient;
        self.hessian -= rhs.hessian;
    }
}

impl<const N: usize> SubAssign<f64> for Hessian<N> {
    #[inline]
    fn sub_assign(&mut self, rhs: f64) {
        self.value -= rhs;
    }
}

impl<const N: usize> Mul<Self> for Hessian<N> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Hessian {
            value: self.value * rhs.value,
            gradient: self.gradient * rhs.value + self.value * rhs.gradient,
            hessian: self.hessian * rhs.value
                + self.value * rhs.hessian
                + self.gradient * rhs.gradient.transpose(),
        }
    }
}

impl<const N: usize> Div<Self> for Hessian<N> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let value = self.value / rhs.value;
        let gradient =
            (self.gradient * rhs.value - self.value * rhs.gradient) / (rhs.value * rhs.value);
        let hessian = self.hessian - gradient * rhs.gradient.transpose() - value * rhs.hessian;
        let hessian = hessian / rhs.value;
        Hessian {
            value,
            gradient,
            hessian,
        }
    }
}

impl<const N: usize> Div<f64> for Hessian<N> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f64) -> Self {
        Hessian {
            value: self.value / rhs,
            gradient: self.gradient / rhs,
            hessian: self.hessian / rhs,
        }
    }
}

impl<const N: usize> Mul<f64> for Hessian<N> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f64) -> Self {
        Hessian {
            value: self.value * rhs,
            gradient: self.gradient * rhs,
            hessian: self.hessian * rhs,
        }
    }
}

impl<const N: usize> MulAssign<Self> for Hessian<N> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.value *= rhs.value;
        self.gradient = self.gradient * rhs.value + self.value * rhs.gradient;
        self.hessian = self.hessian * rhs.value
            + self.value * rhs.hessian
            + self.gradient * rhs.gradient.transpose();
    }
}

impl<const N: usize> MulAssign<f64> for Hessian<N> {
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
        self.value *= rhs;
        self.gradient *= rhs;
        self.hessian *= rhs;
    }
}

// ----------------Operators end------------------

// ----------------num-traits begin

impl<const N: usize> Zero for Hessian<N> {
    #[inline]
    fn zero() -> Self {
        Hessian {
            value: 0.0,
            gradient: SVector::<f64, N>::zeros(),
            hessian: SMatrix::<f64, N, N>::zeros(),
        }
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self.value.is_zero() && self.gradient.is_zero() && self.hessian.is_zero()
    }
}

impl<const N: usize> One for Hessian<N> {
    #[inline]
    fn one() -> Self {
        Hessian {
            value: 1.0,
            gradient: SVector::<f64, N>::zeros(),
            hessian: SMatrix::<f64, N, N>::zeros(),
        }
    }
    #[inline]
    fn is_one(&self) -> bool {
        self.value.is_one() && self.gradient.is_zero() && self.hessian.is_zero()
    }
}

// ----------------num-traits end

// ----------------primitives conversion begin
impl<const N: usize> From<f64> for Hessian<N> {
    #[inline]
    fn from(f: f64) -> Self {
        Hessian {
            value: f,
            gradient: SVector::<f64, N>::zeros(),
            hessian: SMatrix::<f64, N, N>::zeros(),
        }
    }
}

impl<const N: usize> From<Hessian<N>> for f64 {
    #[inline]
    fn from(g: Hessian<N>) -> Self {
        g.value
    }
}

// ----------------primitives conversion end

// ----------------vector constructions begin

pub fn vector_to_hessians<const N: usize>(vec: SVector<f64, N>) -> SVector<Hessian<N>, N> {
    let mut res = SVector::<Hessian<N>, N>::zeros();
    for i in 0..N {
        res[i].value = vec[i];
        res[i].gradient[i] = 1.0;
    }
    res
}

pub fn constant_matrix_to_hessians<const M: usize, const N: usize, const P: usize>(
    vec: SMatrix<f64, M, N>,
) -> SMatrix<Hessian<P>, M, N> {
    let mut res = SMatrix::<Hessian<P>, M, N>::zeros();
    for (i, j) in vec.iter().zip(res.iter_mut()) {
        j.value = *i;
    }
    res
}

pub fn hessians_to_vector<const M: usize, const N: usize, const P: usize>(
    vec: SMatrix<Hessian<P>, M, N>,
) -> SMatrix<f64, M, N> {
    let mut res = SMatrix::<f64, M, N>::zeros();
    for (i, j) in vec.iter().zip(res.iter_mut()) {
        *j = i.value;
    }
    res
}

// ----------------vector constructions end
