use crate::mylog;
use crate::LineSearch;
use crate::LinearSolver;
use crate::Problem;
use crate::Solver;

use na::DVector;
use nalgebra as na;

pub struct NewtonSolverMut {
    pub max_iter: usize,
    pub epi: f64,
}

impl<P: Problem, L: LinearSolver<MatrixType = P::HessianType>, LS: LineSearch<P>> Solver<P, L, LS>
    for NewtonSolverMut
{
    fn epi(&self) -> f64 {
        self.epi
    }

    fn solve<T: std::io::Write>(
        &self,
        p: &mut P,
        lin: &L,
        ls: &LS,
        input: &DVector<f64>,
        #[cfg(feature = "log")] log: &mut T,
    ) -> DVector<f64> {
        let mut g = p.gradient_mut(input).unwrap();

        let mut h: P::HessianType;
        let mut count = 0;
        let mut res = input.clone();
        let mut assem_count: usize;
        // #[cfg(feature = "log")]
        // let mut old_value = p.apply(&input);
        // #[cfg(feature = "log")]
        // let mut new_value: f64;
        while g.norm() > self.epi {
            let _start = std::time::Instant::now();
            let result = p.hessian_mut(&res);
            assem_count = result.1;
            h = result.0.unwrap();

            // mylog!(log, "hessian time spend: ", _start.elapsed().as_secs_f32());
            let _start = std::time::Instant::now();
            let delta = lin.solve(&h, &g);
            // mylog!(
            //     log,
            //     "linear solver time spend: ",
            //     _start.elapsed().as_secs_f32()
            // );
            // mylog!(
            //     log,
            //     "linear residual: ",
            //     (crate::MatrixType::mul(&h, &delta) - &g).norm()
            // );

            let scalar = ls.search(p, &res, &delta);
            // mylog!(log, "line search scalar: ", scalar);
            let delta = delta * scalar;
            res -= &delta;
            g = p.gradient_mut(&res).unwrap();
            if count > self.max_iter {
                panic!("max newtons steps");
            }
            // crate::run_when_logging!(new_value = p.apply(&res));
            // mylog!(log, "delta length", delta.norm());
            // mylog!(log, "value", p.apply(&res));
            // mylog!(log, "value reduction", old_value - new_value);
            // mylog!(log, "gradient norm", g.norm());
            // mylog!(log, " ", " ");
            count += 1;
            // crate::run_when_logging!(old_value = new_value);
        }
        mylog!(log, "newton steps", count);
        res
    }
}
