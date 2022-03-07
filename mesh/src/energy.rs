use std::ops::Mul;

use nalgebra::{ClosedAdd, ClosedMul, ClosedSub, SMatrix, SVector, Scalar};
use num::{One, Zero};

pub trait Energy<const L: usize, const D: usize> {
    fn energy<T>(&self, vec: SVector<T, L>, inv_mat: &SMatrix<T, D, D>, size: f64) -> T
    where
        T: Scalar
            + Zero
            + One
            + ClosedAdd
            + ClosedMul
            + ClosedSub
            + Copy
            + Copy
            + Mul<f64, Output = T>;
}

struct StVenantVirchhoff<const DIM: usize> {
    pub mu: f64,
    pub lambda: f64,
}

impl Energy<6, 2> for StVenantVirchhoff<2> {
    fn energy<T>(&self, vec: SVector<T, 6>, inv_mat: &SMatrix<T, 2, 2>, size: f64) -> T
    where
        T: Scalar + Zero + One + ClosedAdd + ClosedMul + ClosedSub + Copy + Mul<f64, Output = T>,
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
}

impl Energy<12, 3> for StVenantVirchhoff<3> {
    fn energy<T>(&self, vec: SVector<T, 12>, inv_mat: &SMatrix<T, 3, 3>, size: f64) -> T
    where
        T: Scalar + Zero + One + ClosedAdd + ClosedMul + ClosedSub + Copy + Mul<f64, Output = T>,
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
}

struct NeoHookean<const DIM: usize> {
    pub mu: f64,
    pub lambda: f64,
}

impl Energy<6, 2> for NeoHookean<2> {
    fn energy<T>(&self, vec: SVector<T, 6>, inv_mat: &SMatrix<T, 2, 2>, size: f64) -> T
    where
        T: Scalar + Zero + One + ClosedAdd + ClosedMul + ClosedSub + Copy + Mul<f64, Output = T>,
    {
        todo!();
    }
}

// macro_rules! neohookean_2d {
//     ($vec:ident, $ene:ident,$mat:ident,$inv_mat:ident, $square:expr, $type:ty) => {
//         let $mat = na::matrix![
//                 $vec[4]-$vec[0], $vec[2]-$vec[0];
//                 $vec[5]-$vec[1], $vec[3]-$vec[1];
//             ];
//         let matrix_f = $mat*$inv_mat;
//         let i1 = matrix_f.transpose()*matrix_f;
//         let i1 = i1.trace();
//         // let i2=  ?
//         // Is i2 used in formula?
//         let i3 = matrix_f.determinate();
//         let $ene = (i1-<$type>::one()*2.0-i3.log())*(MIU/2.0);
//     };
// }

// macro_rules! st_venant_virchhoff_2d {
//     ($vec:ident, $ene:ident,$mat:ident,$inv_mat:ident, $square:expr, $type:ty) => {
//         let $mat = na::matrix![
//                 $vec[4]-$vec[0], $vec[2]-$vec[0];
//                 $vec[5]-$vec[1], $vec[3]-$vec[1];
//             ];
//         let $mat = $mat*$inv_mat;
//         let $mat  = ($mat.transpose() * $mat  -
//         na::matrix![
//             <$type>::one(), <$type>::zero();
//             <$type>::zero(), <$type>::one();
//         ]) *(<$type>::one()*0.5);

//         let $ene = ($mat.transpose()*$mat).trace()*(<$type>::one()*MIU) +
//          $mat.trace()*$mat.trace()*(<$type>::one()*(0.5*LAMBDA));
//          let $ene = $ene *(<$type>::one()*$square);

//     };
// }

// macro_rules! st_venant_virchhoff_3d {
//     ($vec:ident, $ene:ident,$mat:ident,$inv_mat:ident, $square:expr, $type:ty) => {
//         let $mat = nalgebra::matrix![
//                 $vec[3]-$vec[0], $vec[6]-$vec[0],$vec[9]-$vec[0];
//                 $vec[4]-$vec[1], $vec[7]-$vec[1],$vec[10]-$vec[1];
//                 $vec[5]-$vec[2], $vec[8]-$vec[2],$vec[11]-$vec[2];
//             ];
//         let $mat = $mat*$inv_mat;

//         let $mat  = ($mat.transpose() * $mat  -
//         nalgebra::matrix![
//             <$type>::one(), <$type>::zero(),<$type>::zero();
//             <$type>::zero(), <$type>::one(),<$type>::zero();
//             <$type>::zero(), <$type>::zero(),<$type>::one();
//         ]) *(<$type>::one()*0.5);

//         let $ene = ($mat.transpose()*$mat).trace()*(<$type>::one()*MIU) +
//          $mat.trace()*$mat.trace()*(<$type>::one()*(0.5*LAMBDA));
//          let $ene = $ene *(<$type>::one()*$square);
//     };
// }
