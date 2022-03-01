use crate::LineSearch;
use crate::LinearSolver;
use crate::MatrixType;
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
        while g.norm() > 0.1 {
            h = p.hessian(&res).unwrap();
            let delta = lin.solve(&h, &g);
            println!("ls finish");
            #[cfg(feature = "log")]
            {
                writeln!(log, "linear residual: {},", (&h.mul(&delta) - &g).norm()).unwrap();
            }
            let scalar = ls.search(p, &res, &delta);
            #[cfg(feature = "log")]
            {
                writeln!(log, "line search: {},", scalar).unwrap();
            }
            let delta = delta * scalar;
            res -= &delta;
            g = p.gradient(&res).unwrap();
            if count > self.max_iter {
                break;
            }
            #[cfg(feature = "log")]
            {
                writeln!(log, "delta length: {}", delta.norm()).unwrap();
                writeln!(log, "value: {}", p.apply(&res)).unwrap();
                log.write_all(b"gradient norm").unwrap();
                log.write_all(g.norm().to_string().as_bytes()).unwrap();
                log.write_all(b"\n").unwrap();
                writeln!(log, " ").unwrap();
            }
            count += 1;
        }
        #[cfg(feature = "log")]
        {
            log.write_all(b"newton step: ").unwrap();
            log.write_all(count.to_string().as_bytes()).unwrap();
            log.write_all(b"\n").unwrap();
        }
        res
    }
}
