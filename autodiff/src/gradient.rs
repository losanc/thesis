use na::{SMatrix, SVector};
use nalgebra as na;
use num::{One, Zero};
use std::fmt;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};
#[derive(Clone, PartialEq, Debug, Copy)]
pub struct Gradient<const N: usize> {
    value: f64,
    gradient: SVector<f64, N>,
}

impl<const N: usize> Gradient<N> {
    pub fn gradient(&self) -> SVector<f64, N> {
        self.gradient
    }
    pub fn value(&self) -> f64 {
        self.value
    }
    pub fn to_constant(&mut self) {
        self.gradient = SVector::<f64, N>::zeros();
    }
}

impl<const N: usize> fmt::Display for Gradient<N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

// ----------------Operators begin

impl<const N: usize> Add<Self> for Gradient<N> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Gradient {
            value: self.value + rhs.value,
            gradient: self.gradient + rhs.gradient,
        }
    }
}

impl<const N: usize> Add<f64> for Gradient<N> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: f64) -> Self {
        Gradient {
            value: self.value + rhs,
            gradient: self.gradient,
        }
    }
}

impl<const N: usize> AddAssign<Self> for Gradient<N> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.value += rhs.value;
        self.gradient += rhs.gradient;
    }
}

impl<const N: usize> AddAssign<f64> for Gradient<N> {
    #[inline]
    fn add_assign(&mut self, rhs: f64) {
        self.value += rhs;
    }
}

impl<const N: usize> Sub<Self> for Gradient<N> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Gradient {
            value: self.value - rhs.value,
            gradient: self.gradient - rhs.gradient,
        }
    }
}

impl<const N: usize> Sub<f64> for Gradient<N> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: f64) -> Self {
        Gradient {
            value: self.value - rhs,
            gradient: self.gradient,
        }
    }
}

impl<const N: usize> SubAssign<Self> for Gradient<N> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.value -= rhs.value;
        self.gradient -= rhs.gradient;
    }
}

impl<const N: usize> SubAssign<f64> for Gradient<N> {
    #[inline]
    fn sub_assign(&mut self, rhs: f64) {
        self.value -= rhs;
    }
}

impl<const N: usize> Mul<Self> for Gradient<N> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Gradient {
            value: self.value * rhs.value,
            gradient: self.gradient * rhs.value + self.value * rhs.gradient,
        }
    }
}

impl<const N: usize> Mul<f64> for Gradient<N> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f64) -> Self {
        Gradient {
            value: self.value * rhs,
            gradient: self.gradient * rhs,
        }
    }
}

impl<const N: usize> MulAssign<Self> for Gradient<N> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.value *= rhs.value;
        self.gradient = self.gradient * rhs.value + self.value * rhs.gradient;
    }
}

impl<const N: usize> MulAssign<f64> for Gradient<N> {
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
        self.value *= rhs;
        self.gradient *= rhs;
    }
}

// ----------------Operators end

// ----------------num-traits begin

impl<const N: usize> Zero for Gradient<N> {
    #[inline]
    fn zero() -> Self {
        Gradient {
            value: 0.0,
            gradient: SVector::<f64, N>::zeros(),
        }
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self.value.is_zero() && self.gradient.is_zero()
    }
}

impl<const N: usize> One for Gradient<N> {
    #[inline]
    fn one() -> Self {
        Gradient {
            value: 1.0,
            gradient: SVector::<f64, N>::zeros(),
        }
    }
    #[inline]
    fn is_one(&self) -> bool {
        self.value.is_one() && self.gradient.is_zero()
    }
}

// ----------------num-traits end

// ----------------primitives conversion begin
impl<const N: usize> From<f64> for Gradient<N> {
    #[inline]
    fn from(f: f64) -> Self {
        Gradient {
            value: f,
            gradient: SVector::<f64, N>::zeros(),
        }
    }
}

impl<const N: usize> From<Gradient<N>> for f64 {
    #[inline]
    fn from(g: Gradient<N>) -> Self {
        g.value
    }
}

// ----------------primitives conversion end

// ----------------vector constructions begin

pub fn vector_to_gradients<const N: usize>(vec: &SVector<f64, N>) -> SVector<Gradient<N>, N> {
    let mut res = SVector::<Gradient<N>, N>::zeros();
    for i in 0..N {
        res[i].value = vec[i];
        res[i].gradient[i] = 1.0;
    }
    res
}

pub fn constant_matrix_to_gradients<const M: usize, const N: usize, const P: usize>(
    vec: &SMatrix<f64, M, N>,
) -> SMatrix<Gradient<P>, M, N> {
    let mut res = SMatrix::<Gradient<P>, M, N>::zeros();
    for (i, j) in vec.iter().zip(res.iter_mut()) {
        j.value = *i;
    }
    res
}

pub fn gradients_to_vector<const M: usize, const N: usize, const P: usize>(
    vec: &SMatrix<Gradient<P>, M, N>,
) -> SMatrix<f64, M, N> {
    let mut res = SMatrix::<f64, M, N>::zeros();
    for (i, j) in vec.iter().zip(res.iter_mut()) {
        *j = i.value;
    }
    res
}

// ----------------vector constructions end
