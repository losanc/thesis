use crate::LineSearch;
use crate::LinearSolver;
use crate::Problem;
use crate::Solver;

use na::DVector;
use nalgebra as na;

pub struct NewtonSolver {
    pub max_iter: usize,
}

impl<P: Problem, L: LinearSolver<MatrixType = P::HessianType>, LS: LineSearch<P>> Solver<P, L, LS>
    for NewtonSolver
{
    fn solve<T: std::io::Write>(&self, p: &P, lin: &L, ls: &LS, input: &DVector<f64>,log: &mut T) -> DVector<f64>{
        let mut g = p.gradient(input).unwrap();
        let mut h: P::HessianType;
        let mut count = 0;
        let mut res = input.clone();
        while g.norm() > 0.1 {
            h = p.hessian(&res).unwrap();
            let delta = lin.solve(&h, &g);
            let delta = ls.search(p, &res, delta);
            res -= delta;
            g = p.gradient(&res).unwrap();
            if count > self.max_iter {
                break;
            }
            // log.write_all(b"gradient norm");
            // log.write_all(g.norm().to_string().as_bytes());
            // log.write_all(b"\n");
            count += 1;
        }
        // log.write_all(b"newton step: ");
        // log.write_all(count.to_string().as_bytes());
        // log.write_all(b"\n");
        res
    }
}
