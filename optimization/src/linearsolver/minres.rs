use std::marker::PhantomData;

use nalgebra::DVector;

use crate::{LinearSolver, MatrixType};

pub struct MINRESLinear<T: MatrixType> {
    phantom: PhantomData<T>,
    pub epi: f64,
}

impl<T: MatrixType> LinearSolver for MINRESLinear<T> {
    type MatrixType = T;
    fn new() -> Self {
        MINRESLinear {
            phantom: PhantomData,
            epi: 0.001,
        }
    }
    #[allow(non_snake_case)]
    fn solve(&self, A: &Self::MatrixType, rhs: &DVector<f64>) -> DVector<f64> {
        let tol = rhs.norm();
        let mut c = 1.0;
        let mut c_old = 1.0;
        let mut s = 0.0;
        let mut s_old = 0.0;
        let mut eta = 1.0;

        let mut x = DVector::<f64>::zeros(rhs.len());

        let mut v_new;
        let mut v_old; 
        let mut v = DVector::<f64>::zeros(rhs.len());
        let mut p_old = DVector::<f64>::zeros(rhs.len());
        let mut p_oold;
        let mut p = DVector::<f64>::zeros(rhs.len());

        v_new = rhs - A.mul(&x);

        let mut res_norm = v_new.norm();
        let mut beta_new = res_norm;
        let beta_one = beta_new;

        v_new *= 1.0 / beta_new;

        for _ in 0..10000000 {
            let beta = beta_new;
            v_old = v.clone();
            v = v_new.clone();

            v_new = A.mul(&v);

            let alpha = v.transpose() * &v_new;
            let alpha = alpha[(0, 0)];
            v_new -= &v_old * beta;
            v_new -= &v * alpha;
            beta_new = v_new.norm();
            v_new *= 1.0 / beta_new;

            let r3 = s_old * beta; // s, s_old, c and c_old are still from previous iteration
            let tr = c_old * beta;
            let r2 = alpha * s + c * tr;
            let r1_hat = c * alpha - tr * s;

            let r1_inv = 1.0 / (r1_hat * r1_hat + beta_new * beta_new).sqrt();

            c_old = c; // store for next iteration
            s_old = s; // store for next iteration

            // [ c  s ]
            // [-s  c ]
            c = r1_hat * r1_inv; // new cosine
            s = beta_new * r1_inv; // new sine

            p_oold = p_old;
            p_old = p;
            p = v.clone();
            p -= r2 * &p_old;
            p -= r3 * &p_oold;
            p *= r1_inv;
            x += &p * beta_one * c * eta;
            res_norm *= s.abs();
            if res_norm < self.epi * tol {
                return x;
            }
            eta *= -s;
        }
        println!("{res_norm}");
        panic!("shouldn't");

        // x
    }
}

// #[test]
// fn test() {
//     use nalgebra::{dvector, matrix};
//     let a = matrix![
//         1.0,3.0,4.5,5.0,7.8;
//         33.0,25.0,4.0,5.0,7.98;
//         1.5,3.6,4.5,9.0,78.8;
//         3.0,5.0,47.0,54.0,31.498;
//         121.5,34.6,4.845,9.0,42.8;
//     ];
//     let a = a * a.transpose();
//     let sparse_a = CsrMatrix::from(&a);
//     let b = dvector![1.0, 2.0, 3.0, 4.0, 5.0];
//     let solver = MINRESLinear {};
//     let res = solver.solve(&sparse_a, &b);
//     let real_res = a.full_piv_lu().solve(&b).unwrap();
//     println!(" {}, \n", a * res - b);
// }
