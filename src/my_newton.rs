use nalgebra::DVector;
use optimization::{LineSearch, LinearSolver, Problem, Solver, MatrixType};

pub trait MyProblem: Problem {
    fn my_gradient(&self, x: &DVector<f64>) -> (Option<DVector<f64>>, Option<Vec<usize>>) {
        (self.gradient(x), None)
    }

    fn my_hessian(
        &self,
        x: &DVector<f64>,
        _active_set: &[usize],
    ) -> Option<<Self as Problem>::HessianType> {
        self.hessian(x)
    }
}

pub struct MyNewtonSolver {
    pub max_iter: usize,
}

impl<P: MyProblem, L: LinearSolver<MatrixType = P::HessianType>, LS: LineSearch<P>> Solver<P, L, LS>
    for MyNewtonSolver
{
    fn solve<T: std::io::Write>(
        &self,
        p: &P,
        lin: &L,
        ls: &LS,
        input: &DVector<f64>,
        log: &mut T,
    ) -> DVector<f64> {
        let (g, active_set) = p.my_gradient(input);
        let mut g = g.unwrap();
        let mut active_set = active_set.unwrap();
        let mut h: P::HessianType;
        let mut count = 0;
        let mut res = input.clone();
        while g.norm() > 1e-5 {
            h = p.my_hessian(&res, &active_set).unwrap();
            let delta = lin.solve(&h, &g);
            #[cfg(feature = "log")]
            {
                writeln!(log, "linear residual: {},", (&h.mul(&delta)-&g).norm()).unwrap();
            }
            let scalar = ls.search(p, &res, &delta);
            #[cfg(feature = "log")]
            {
                writeln!(log, "line search: {},", scalar).unwrap();
            }
            let delta = delta*scalar;
            res -=  &delta;
            let (t1, t2) = p.my_gradient(&res);
            g = t1.unwrap();
            active_set = t2.unwrap();
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
            log.write_all(b"newton step: ").expect("io error");
            log.write_all(count.to_string().as_bytes())
                .expect("io error");
            log.write_all(b"\n").expect("io error");
        }
        res
    }
}
