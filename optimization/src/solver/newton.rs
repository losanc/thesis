use crate::LineSearch;
use crate::LinearSolver;
use crate::MatrixType;
use crate::Problem;
use crate::Solver;

use na::DVector;
use nalgebra as na;

pub struct NewtonSolver {}

impl<P: Problem, L: LinearSolver<MatrixType = P::HessianType>, LS: LineSearch<P>> Solver<P, L, LS>
    for NewtonSolver
{
    fn solve(&self, p: &P, lin: &L, ls: &LS, input: &DVector<f64>) -> DVector<f64> {
        let mut g = p.gradient(input).unwrap();
        let mut h: P::HessianType;
        let mut count = 0;
        let mut res = input.clone();
        while g.norm() > 0.1 {
            h = p.hessian(&res).unwrap();
            print!("gradient norm{}\n", g.norm());
            let delta = lin.solve(&h, &g);
            let delta = ls.search(p, &res, delta);
            res -= delta;
            g = p.gradient(&res).unwrap();
            count += 1;
        }
        println!("newton step: {}\n", count);
        res
    }
}
