use crate::LinearSolver;
use crate::MatrixType;
use na::DVector;
use nalgebra as na;
use std::marker::PhantomData;

pub trait PreConditioner {
    type MatrixType: crate::MatrixType;
    fn get(mat: &Self::MatrixType) -> Self::MatrixType;
    fn new() -> Self;
}

pub struct JacobianPre<T: MatrixType> {
    phantom: PhantomData<T>,
}

impl<T: MatrixType> PreConditioner for JacobianPre<T> {
    type MatrixType = T;
    fn get(mat: &T) -> T {
        mat.inverse_diagoanl()
    }
    fn new() -> Self {
        JacobianPre {
            phantom: PhantomData,
        }
    }
}

pub struct NoPre<T: MatrixType> {
    phantom: PhantomData<T>,
}

impl<T: MatrixType> PreConditioner for NoPre<T> {
    type MatrixType = T;
    fn get(mat: &T) -> T {
        mat.identity()
    }
    fn new() -> Self {
        NoPre {
            phantom: PhantomData,
        }
    }
}

pub struct NewtonCG<P: PreConditioner> {
    // it has no use, but mark the type P
    phantom: PhantomData<P>,
}

#[allow(non_snake_case, clippy::many_single_char_names)]
impl<P: PreConditioner> LinearSolver for NewtonCG<P> {
    type MatrixType = P::MatrixType;

    fn new() -> Self {
        NewtonCG {
            phantom: PhantomData,
        }
    }

    fn solve(&self, A: &Self::MatrixType, b: &DVector<f64>) -> DVector<f64> {
        let m_inv = P::get(A);
        let mut x = DVector::<f64>::zeros(b.nrows());
        let mut r = b - A.mul(&x);
        if r.norm() < 0.1 {
            return x;
        }
        let mut z = m_inv.mul(&r);
        let mut p = z.clone();
        let mut alpha: f64;
        let mut beta: f64;

        let mut numerate = (r.transpose() * &z)[(0, 0)];
        for i in 0..10000 {
            // println!("{}",i);
            let denominator = p.transpose() * (A.mul(&p));
            let denominator = denominator[(0, 0)];
            if denominator < 0.0 {
                if i == 0 {
                    return b.clone();
                } else {
                    return x;
                }
            }
            alpha = numerate / denominator;
            x += alpha * &p;
            r -= alpha * (A.mul(&p));

            if r.norm() < 0.001 {
                println!("lin {}", i);
                return x;
            }
            z = m_inv.mul(&r);
            let new_num = (r.transpose() * &z)[(0, 0)];
            beta = new_num / numerate;

            p = &z + beta * &p;
            numerate = new_num;
        }
        x
    }
}
