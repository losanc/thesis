use na::DVector;
use nalgebra as na;
use optimization::LineSearch;
use optimization::LinearSolver;
use optimization::{Problem, Solver};
use std::io::Write;

pub trait ScenarioProblem: Problem {
    fn frame_init(&mut self);
    fn initial_guess(&self) -> DVector<f64>;

    fn set_all_vertices_vector(&mut self, vertices: DVector<f64>);
    fn save_to_file(&self, frame: usize);
    fn frame_end(&mut self);
}

pub struct Scenario<
    P: ScenarioProblem,
    S: Solver<P, LSo, LSe>,
    LSo: LinearSolver<MatrixType = P::HessianType>,
    LSe: LineSearch<P>,
> {
    problem: P,
    frame: usize,
    solver: S,
    linearsolver: LSo,
    ls: LSe,
    #[cfg(feature = "log")]
    file1: std::fs::File,
    #[cfg(not(feature = "log"))]
    file1: Vec<u8>,
}

impl<P, S, LSo, LSe> Scenario<P, S, LSo, LSe>
where
    P: ScenarioProblem,
    S: Solver<P, LSo, LSe>,
    LSo: LinearSolver<MatrixType = P::HessianType>,
    LSe: LineSearch<P>,
{
    pub fn new(p: P, s: S, lso: LSo, lse: LSe, filename: &str, comment: &str) -> Self {
        let mut file1;
        #[cfg(feature = "log")]
        {
            file1 = std::fs::File::create(filename).unwrap();
            writeln!(file1, "{}", comment).unwrap();
        }
        #[cfg(not(feature = "log"))]
        {
            file1 = Vec::<u8>::new();
        }

        Scenario {
            problem: p,
            frame: 0,
            solver: s,
            linearsolver: lso,
            ls: lse,
            file1,
        }
    }

    pub fn step(&mut self) {
        self.problem.frame_init();
        let initial_guess = self.problem.initial_guess();
        // let start = std::time::Instant::now();
        #[cfg(feature = "log")]
        {
            writeln!(self.file1, "\n\nFrame: {}", self.frame).unwrap();
        }
        let res2 = self.solver.solve(
            &mut self.problem,
            &self.linearsolver,
            &self.ls,
            &initial_guess,
            &mut self.file1,
        );
        // #[cfg(feature = "log")]
        // {
        //     let duration = start.elapsed();
        //     writeln!(self.file1, "Duration: {}\n\n", duration.as_secs_f32()).unwrap();
        // }
        self.problem.set_all_vertices_vector(res2);
        self.frame += 1;
        self.problem.frame_end();
        #[cfg(feature = "save")]
        self.problem.save_to_file(self.frame);
    }
}
