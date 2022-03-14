use crate::mylog;
use crate::LineSearch;
use crate::LinearSolver;
use crate::MatrixType;
use crate::Problem;
use crate::Solver;

use na::DVector;
use nalgebra as na;

pub struct NewtonSolver {
    pub max_iter: usize,
    pub epi: f64,
}

impl<P: Problem, L: LinearSolver<MatrixType = P::HessianType>, LS: LineSearch<P>> Solver<P, L, LS>
    for NewtonSolver
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
        let mut g = p.gradient(input).unwrap();

        let mut h: P::HessianType;
        let mut count = 0;
        let mut res = input.clone();
        let mut old_value = p.apply(&input);
        let mut new_value: f64;
        while g.norm() > self.epi {
            h = p.hessian(&res).unwrap();
            println!("finished hessian");
            let delta = lin.solve(&h, &g);
            println!("finished linear solver");
            mylog!(log, "linear residual: ", (&h.mul(&delta) - &g).norm());
            let scalar = ls.search(p, &res, &delta);
            mylog!(log, "line search scalar: ", scalar);
            let delta = delta * scalar;
            res -= &delta;
            g = p.gradient(&res).unwrap();
            if count > self.max_iter {
                break;
            }
            new_value = p.apply(&res);
            mylog!(log, "delta length", delta.norm());
            mylog!(log, "value", p.apply(&res));
            mylog!(log, "value reduction", old_value - new_value);
            mylog!(log, "gradient norm", g.norm());
            mylog!(log, " ", " ");
            count += 1;
            old_value = new_value;
        }
        mylog!(log, "newton stesp", count);
        res
    }
}
