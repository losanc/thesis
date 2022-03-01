use na::DVector;
use nalgebra as na;
use optimization::LineSearch;
use optimization::LinearSolver;
use optimization::{Problem, Solver};
use std::io::Write;
use std::time;

use crate::my_newton::MyNewtonSolver;
use crate::my_newton::MyProblem;
use crate::mylog;

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

    pub fn mystep(&mut self, use_as_result: bool) {
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
        let start = time::Instant::now();
        let _res2 = self.mysolver.solve(
            &self.problem,
            &self.linearsolver,
            &self.ls,
            &initial_guess,
            &mut self.file1,
        );
        let duration = start.elapsed();
        mylog!(self.file1, "time elapsed ", duration.as_secs_f32());
        if use_as_result {
            self.frame += 1;
            self.problem.set_all_vertices_vector(_res2);

            #[cfg(feature = "save")]
            self.problem.save_to_file(self.frame);
        }
    }

    pub fn step(&mut self, use_as_result: bool) {
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
        let start = time::Instant::now();
        let res2 = self.solver.solve(
            &self.problem,
            &self.linearsolver,
            &self.ls,
            &initial_guess,
            &mut self.file2,
        );
        let duration = start.elapsed();
        mylog!(self.file2, "time elapsed ", duration.as_secs_f32());
        if use_as_result {
            self.problem.set_all_vertices_vector(res2);

            self.frame += 1;
            self.problem.frame_end();
            #[cfg(feature = "save")]
            self.problem.save_to_file(self.frame);
        }
    }
}
