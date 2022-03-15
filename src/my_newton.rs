use nalgebra::DVector;
use optimization::{LineSearch, LinearSolver, MatrixType, Problem, Solver};

use crate::mylog;
pub trait MyProblem: Problem {
    fn my_gradient(
        &self,
        x: &DVector<f64>,
    ) -> (
        Option<DVector<f64>>,
        Option<std::collections::HashSet<usize>>,
    ) {
        (self.gradient(x), None)
    }

    fn my_hessian(
        &self,
        x: &DVector<f64>,
        _active_set: &std::collections::HashSet<usize>,
    ) -> Option<<Self as Problem>::HessianType> {
        self.hessian(x)
    }
}

pub struct MyNewtonSolver {
    pub max_iter: usize,
    pub epi: f64,
}

impl<P: MyProblem, L: LinearSolver<MatrixType = P::HessianType>, LS: LineSearch<P>> Solver<P, L, LS>
    for MyNewtonSolver
{
    fn epi(&self) -> f64 {
        self.epi
    }
    fn solve<T: std::io::Write>(
        &self,
        p: &P,
        lin: &L,
        ls: &LS,
        input: &DVector<f64>,
        log: &mut T,
    ) -> DVector<f64> {
        let (g, ase) = p.my_gradient(input);
        let mut g = g.unwrap();
        let mut active_set = ase.unwrap();
        let mut h: P::HessianType;
        let mut count = 0;
        let mut res = input.clone();
        let mut old_value = p.apply(&input);
        let mut new_value: f64;
        while g.norm() > self.epi {
            h = p.my_hessian(&res, &active_set).unwrap();
            let delta = lin.solve(&h, &g);
            mylog!(log, "linear residual: ", (&h.mul(&delta) - &g).norm());
            let scalar = ls.search(p, &res, &delta);
            mylog!(log, "line search scalar: ", scalar);
            let delta = delta * scalar;
            res -= &delta;
            let (t1, t2) = p.my_gradient(&res);
            g = t1.unwrap();
            active_set = t2.unwrap();
            if count > self.max_iter {
                break;
            }

            new_value = p.apply(&res);
            mylog!(log, "delta length", delta.norm());
            mylog!(log, "value", p.apply(&res));
            mylog!(log, "value reduction", old_value - new_value);
            mylog!(log, "gradient norm", g.norm());
            mylog!(log, " ", " ");
            old_value = new_value;
            count += 1;
        }

        mylog!(log, "newton steps", count);

        res
    }
}
