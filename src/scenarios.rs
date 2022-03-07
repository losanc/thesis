use na::DVector;
use nalgebra as na;
use optimization::LineSearch;
use optimization::LinearSolver;
use optimization::{Problem, Solver};
use std::cell::RefCell;
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
    #[cfg(feature = "log")]
    file1: std::fs::File,
    #[cfg(not(feature = "log"))]
    file1: Vec<u8>,
    #[cfg(feature = "log")]
    file2: std::fs::File,
    #[cfg(not(feature = "log"))]
    file2: Vec<u8>,
}

impl<P, S, LSo, LSe> Scenario<P, S, LSo, LSe>
where
    P: ScenarioProblem + MyProblem,
    S: Solver<P, LSo, LSe>,
    LSo: LinearSolver<MatrixType = P::HessianType>,
    LSe: LineSearch<P>,
{
    pub fn new(p: P, s: S, lso: LSo, lse: LSe) -> Self {
        let mut file1;
        let mut file2;
        #[cfg(feature = "log")]
        {
            file1 = std::fs::File::create("new.txt").unwrap();
            writeln!(file1, "modified newton").unwrap();
            file2 = std::fs::File::create("old.txt").unwrap();
            writeln!(file2, "complete newton").unwrap();
        }
        #[cfg(not(feature = "log"))]
        {
            file1 = Vec::<u8>::new();
            file2 = Vec::<u8>::new();
        }

        Scenario {
            problem: p,
            frame: 0,
            mysolver: MyNewtonSolver {
                max_iter: 30,
                epi: s.epi(),
                frame: RefCell::<usize>::new(0),
            },
            solver: s,
            linearsolver: lso,
            ls: lse,
            file1,
            file2,
        }
    }

    pub fn mystep(&mut self, use_as_result: bool) {
        mylog!(self.file1, "\n\nFrame {}", self.frame);

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
        mylog!(self.file2, "\n\nFrame {}", self.frame);

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
