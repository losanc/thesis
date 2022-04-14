use crate::mylog;
use crate::LineSearch;
use crate::LinearSolver;
use crate::Problem;
use crate::Solver;

use na::DMatrix;
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
        p: &mut P,
        lin: &L,
        ls: &LS,
        input: &DVector<f64>,
        log: &mut T,
    ) -> DVector<f64> {
        let mut g = p.gradient(input).unwrap();

        let mut h: P::HessianType;
        let mut count = 0;
        let mut res = input.clone();
        // let mut old_value = p.apply(&input);
        // let mut new_value: f64;
        while g.norm() > self.epi {
            // let start = std::time::Instant::now();
            h = p.hessian(&res).unwrap();
            // let duration = start.elapsed();
            // mylog!(log, "hessian time spend: ", duration.as_secs_f32());

            // let start = std::time::Instant::now();
            let delta = lin.solve(&h, &g);
            // let duration = start.elapsed();
            // mylog!(log, "linear solver time spend: ", duration.as_secs_f32());
            // mylog!(log, "linear residual: ", (&h.mul(&delta) - &g).norm());
            let scalar = ls.search(p, &res, &delta);
            // mylog!(log, "line search scalar: ", scalar);
            let delta = delta * scalar;
            res -= &delta;
            g = p.gradient(&res).unwrap();
            if count > self.max_iter {
                break;
            }
            // new_value = p.apply(&res);
            // mylog!(log, "delta length", delta.norm());
            // mylog!(log, "value", p.apply(&res));
            // mylog!(log, "value reduction", old_value - new_value);
            // mylog!(log, "gradient norm", g.norm());
            // mylog!(log, " ", " ");
            count += 1;
            // old_value = new_value;
        }
        mylog!(log, "newton stesp", count);
        res
    }
}

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
        log: &mut T,
    ) -> DVector<f64> {
        let mut g = p.gradient_mut(input).unwrap();

        let mut h: P::HessianType;
        let mut count = 0;
        let mut res = input.clone();
        // let mut old_value = p.apply(&input);
        // let mut new_value: f64;
        while g.norm() > self.epi {
            // let start = std::time::Instant::now();
            h = p.hessian_mut(&res).unwrap();
            // let duration = start.elapsed();
            // mylog!(log, "hessian time spend: ", duration.as_secs_f32());

            // let start = std::time::Instant::now();
            let delta = lin.solve(&h, &g);
            // let duration = start.elapsed();
            // mylog!(log, "linear solver time spend: ", duration.as_secs_f32());
            // mylog!(log, "linear residual: ", (&h.mul(&delta) - &g).norm());
            let scalar = ls.search(p, &res, &delta);
            // mylog!(log, "line search scalar: ", scalar);
            let delta = delta * scalar;
            res -= &delta;
            g = p.gradient_mut(&res).unwrap();
            if count > self.max_iter {
                break;
            }
            // new_value = p.apply(&res);
            // mylog!(log, "delta length", delta.norm());
            // mylog!(log, "value", p.apply(&res));
            // mylog!(log, "value reduction", old_value - new_value);
            // mylog!(log, "gradient norm", g.norm());
            // mylog!(log, " ", " ");
            count += 1;
            // old_value = new_value;
        }
        mylog!(log, "newton stesp", count);
        res
    }
}

pub struct NewtonInverseSolver {
    pub max_iter: usize,
    pub epi: f64,
}

impl<P: Problem, L: LinearSolver<MatrixType = P::HessianType>, LS: LineSearch<P>> Solver<P, L, LS>
    for NewtonInverseSolver
{
    fn epi(&self) -> f64 {
        self.epi
    }
    fn solve<T: std::io::Write>(
        &self,
        p: &mut P,
        _lin: &L,
        ls: &LS,
        input: &DVector<f64>,
        log: &mut T,
    ) -> DVector<f64> {
        let mut g = p.gradient_mut(input).unwrap();

        let mut h_inv: DMatrix<f64>;
        let mut count = 0;
        let mut res = input.clone();
        // let mut old_value = p.apply(&input);
        // let mut new_value: f64;
        while g.norm() > self.epi {
            // let start = std::time::Instant::now();
            h_inv = p.hessian_inverse_mut(&res).unwrap();
            let delta = &h_inv * &g;
            // let duration = start.elapsed();
            // mylog!(log, "linear solver time spend: ", duration.as_secs_f32());
            let scalar = ls.search(p, &res, &delta);
            // mylog!(log, "line search scalar: ", scalar);
            let delta = delta * scalar;
            res -= &delta;
            g = p.gradient_mut(&res).unwrap();
            if count > self.max_iter {
                break;
            }
            // new_value = p.apply(&res);
            // mylog!(log, "delta length", delta.norm());
            // mylog!(log, "value", p.apply(&res));
            // mylog!(log, "value reduction", old_value - new_value);
            // mylog!(log, "gradient norm", g.norm());
            // mylog!(log, " ", " ");
            count += 1;
            // old_value = new_value;
        }
        mylog!(log, "newton stesp", count);
        res
    }
}
