use na::DVector;
use nalgebra as na;
use optimization::LineSearch;
use optimization::LinearSolver;
use optimization::{Problem, Solver};
use std::io::Write;

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

pub trait ScenarioProblem: Problem {
    fn frame_init(&mut self);
    fn initial_guess(&self) -> DVector<f64>;

    fn set_all_vertices_vector(&mut self, vertices: DVector<f64>);
    fn save_to_file(&self, frame: usize);
    fn frame_end(&mut self);
}

pub struct Scenario<
    P: ScenarioProblem + MyProblem,
    S: Solver<P, LSo, LSe>,
    LSo: LinearSolver<MatrixType = P::HessianType>,
    LSe: LineSearch<P>,
> {
    problem: P,
    frame: usize,
    solver: S,
    mysolver: MyNewtonSolver,
    linearsolver: LSo,
    ls: LSe,
    file1: std::fs::File,
    file2: std::fs::File,
}

impl<P, S, LSo, LSe> Scenario<P, S, LSo, LSe>
where
    P: ScenarioProblem + MyProblem,
    S: Solver<P, LSo, LSe>,
    LSo: LinearSolver<MatrixType = P::HessianType>,
    LSe: LineSearch<P>,
{
    pub fn new(p: P, s: S, lso: LSo, lse: LSe) -> Self {
        let mut file1 = std::fs::File::create("new.txt").unwrap();
        file1.write_all(b"modified newton\n").expect("io error");
        let mut file2 = std::fs::File::create("old.txt").unwrap();
        file2.write_all(b"complete newton\n").expect("io error");
        Scenario {
            problem: p,
            frame: 0,
            solver: s,
            mysolver: MyNewtonSolver { max_iter: 30 },
            linearsolver: lso,
            ls: lse,
            file1,
            file2,
        }
    }

    pub fn mystep(&mut self) {
        #[cfg(feature = "log")]
        {
            self.file1.write_all(b"\n\nFrame:  ").expect("io error");
            self.file1
                .write_all(self.frame.to_string().as_bytes())
                .expect("io error");
            self.file1.write_all(b"\n").expect("io error");
        }

        self.problem.frame_init();
        let initial_guess = self.problem.initial_guess();

        let _res2 = self.mysolver.solve(
            &self.problem,
            &self.linearsolver,
            &self.ls,
            &initial_guess,
            &mut self.file1,
        );
        // self.frame += 1;
        // self.problem.set_all_vertices_vector(_res2);

        // #[cfg(feature = "save")]
        // self.problem.save_to_file(self.frame);
    }

    pub fn step(&mut self) {
        #[cfg(feature = "log")]
        {
            self.file2.write_all(b"\n\nFrame:  ").expect("io error");
            self.file2
                .write_all(self.frame.to_string().as_bytes())
                .expect("io error");
            self.file2.write_all(b"\n").expect("io error");
        }

        self.problem.frame_init();
        let initial_guess = self.problem.initial_guess();
        let res2 = self.solver.solve(
            &self.problem,
            &self.linearsolver,
            &self.ls,
            &initial_guess,
            &mut self.file2,
        );
        self.problem.set_all_vertices_vector(res2);

        self.frame += 1;
        self.problem.frame_end();
        #[cfg(feature = "save")]
        self.problem.save_to_file(self.frame);
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
        while g.norm() > 1e-4 {
            h = p.my_hessian(&res, &active_set).unwrap();
            let delta = lin.solve(&h, &g);
            let scalar = ls.search(p, &res, &delta);
            res -= scalar * delta;
            let (t1, t2) = p.my_gradient(&res);
            g = t1.unwrap();
            active_set = t2.unwrap();
            if count > self.max_iter {
                break;
            }
            #[cfg(feature = "log")]
            {
                log.write_all(b"gradient norm").expect("io error");
                log.write_all(g.norm().to_string().as_bytes())
                    .expect("io error");
                log.write_all(b"\n").expect("io error");
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
