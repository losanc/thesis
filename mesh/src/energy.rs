use std::ops::{Mul, Sub};

use autodiff::MyLog;
use nalgebra::{ClosedAdd, ClosedMul, ClosedSub, SMatrix, SVector, Scalar};
use num::{One, Zero};

pub trait Energy<const L: usize, const D: usize> {
    fn energy<T>(&self, vec: SVector<T, L>, inv_mat: SMatrix<T, D, D>, size: f64) -> T
    where
        T: Scalar
            + Zero
            + One
            + ClosedAdd
            + ClosedMul
            + ClosedSub
            + Copy
            + Copy
            + MyLog
            + Mul<f64, Output = T>
            + Sub<f64, Output = T>
            + std::fmt::Display;
    fn mu(&self) -> f64;
    fn lambda(&self) -> f64;
}

pub struct StVenantVirchhoff<const DIM: usize> {
    pub mu: f64,
    pub lambda: f64,
}

impl Energy<6, 2> for StVenantVirchhoff<2> {
    fn energy<T>(&self, vec: SVector<T, 6>, inv_mat: SMatrix<T, 2, 2>, size: f64) -> T
    where
        T: Scalar
            + Zero
            + One
            + ClosedAdd
            + ClosedMul
            + ClosedSub
            + Copy
            + MyLog
            + Mul<f64, Output = T>
            + Sub<f64, Output = T>,
    {
        let mat = nalgebra::matrix![
            vec[4]-vec[0], vec[2]-vec[0];
            vec[5]-vec[1], vec[3]-vec[1];
        ];
        let mat = &mat * inv_mat;
        let mat = (mat.transpose() * mat
            - nalgebra::matrix![
                T::one(), T::zero();
                T::zero(), T::one();
            ])
            * (T::one() * 0.5);
        let ene = (mat.transpose() * mat).trace() * (T::one() * self.mu)
            + mat.trace() * mat.trace() * (T::one() * (0.5 * self.lambda));
        let ene = ene * (T::one() * size);
        ene
    }

    fn mu(&self) -> f64 {
        self.mu
    }

    fn lambda(&self) -> f64 {
        self.lambda
    }
}

impl Energy<12, 3> for StVenantVirchhoff<3> {
    fn energy<T>(&self, vec: SVector<T, 12>, inv_mat: SMatrix<T, 3, 3>, size: f64) -> T
    where
        T: Scalar
            + Zero
            + One
            + ClosedAdd
            + ClosedMul
            + ClosedSub
            + Copy
            + MyLog
            + Mul<f64, Output = T>
            + Sub<f64, Output = T>,
    {
        let mat = nalgebra::matrix![
            vec[3]-vec[0], vec[6]-vec[0],vec[9]-vec[0];
            vec[4]-vec[1], vec[7]-vec[1],vec[10]-vec[1];
            vec[5]-vec[2], vec[8]-vec[2],vec[11]-vec[2];
        ];
        let mat = &mat * inv_mat;
        let mat = (mat.transpose() * mat
            - nalgebra::matrix![
                T::one(), T::zero(), T::zero();
                T::zero(), T::one() ,T::zero();
                T::zero(), T::zero() ,T::one();
            ])
            * (T::one() * 0.5);
        let ene = (mat.transpose() * mat).trace() * (T::one() * self.mu)
            + mat.trace() * mat.trace() * (T::one() * (0.5 * self.lambda));
        let ene = ene * (T::one() * size);
        ene
    }

    fn mu(&self) -> f64 {
        self.mu
    }

    fn lambda(&self) -> f64 {
        self.lambda
    }
}
#[derive(Debug)]
pub struct NeoHookean<const DIM: usize> {
    pub mu: f64,
    pub lambda: f64,
}

impl Energy<6, 2> for NeoHookean<2> {
    fn energy<T>(&self, vec: SVector<T, 6>, inv_mat: SMatrix<T, 2, 2>, size: f64) -> T
    where
        T: Scalar
            + Zero
            + One
            + ClosedAdd
            + ClosedMul
            + ClosedSub
            + Copy
            + MyLog
            + Mul<f64, Output = T>
            + Sub<f64, Output = T>
            + std::fmt::Display,
    {
        let mat = nalgebra::matrix![
            vec[4]-vec[0], vec[2]-vec[0];
            vec[5]-vec[1], vec[3]-vec[1];
        ];
        let matrix_f = mat * inv_mat;
        let i1 = matrix_f.transpose() * matrix_f;
        let i1 = i1.trace();
        // let i2=  ?`
        // Is i2 used in formula?

        // i3 = matrix_f.determinate()
        let i3 = matrix_f[(0, 0)] * matrix_f[(1, 1)] - matrix_f[(0, 1)] * matrix_f[(1, 0)];

        // invertion test
        // assert!(i3 > 0.0)
        let i3 = i3 * i3;

        let logi3 = i3.myln();
        let ene = (i1 - logi3 - 2.0) * (self.mu / 2.0) + logi3 * logi3 * (self.lambda / 8.0);
        let ene = ene * (T::one() * size);
        ene
    }

    fn mu(&self) -> f64 {
        self.mu
    }

    fn lambda(&self) -> f64 {
        self.lambda
    }
}

impl Energy<12, 3> for NeoHookean<3> {
    fn energy<T>(&self, vec: SVector<T, 12>, inv_mat: SMatrix<T, 3, 3>, size: f64) -> T
    where
        T: Scalar
            + Zero
            + One
            + ClosedAdd
            + ClosedMul
            + ClosedSub
            + Copy
            + MyLog
            + Mul<f64, Output = T>
            + Sub<f64, Output = T>,
    {
        let mat = nalgebra::matrix![
            vec[3]-vec[0], vec[6]-vec[0],vec[9]-vec[0];
            vec[4]-vec[1], vec[7]-vec[1],vec[10]-vec[1];
            vec[5]-vec[2], vec[8]-vec[2],vec[11]-vec[2];
        ];
        let matrix_f = mat * inv_mat;
        let i1 = matrix_f.transpose() * matrix_f;
        let i1 = i1.trace();
        // let i2=  ?`
        // Is i2 used in formula?

        // i3 = matrix_f.determinate()
        let i3 = matrix_f[(0, 0)] * matrix_f[(1, 1)] * matrix_f[(2, 2)]
            + matrix_f[(0, 1)] * matrix_f[(1, 2)] * matrix_f[(2, 0)]
            + matrix_f[(0, 2)] * matrix_f[(1, 0)] * matrix_f[(2, 1)]
            - matrix_f[(0, 2)] * matrix_f[(1, 1)] * matrix_f[(2, 0)]
            - matrix_f[(0, 1)] * matrix_f[(1, 0)] * matrix_f[(2, 2)]
            - matrix_f[(0, 0)] * matrix_f[(2, 1)] * matrix_f[(1, 2)];
        let i3 = i3 * i3;
        let logi3 = i3.myln();
        let ene = (i1 - logi3 - 3.0) * (self.mu / 2.0) + logi3 * logi3 * (self.lambda / 8.0);
        let ene = ene * (T::one() * size);
        ene
    }

    fn mu(&self) -> f64 {
        self.mu
    }

    fn lambda(&self) -> f64 {
        self.lambda
    }
}
