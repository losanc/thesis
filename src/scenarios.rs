use na::DVector;
use nalgebra as na;
use optimization::LineSearch;
use optimization::LinearSolver;
use optimization::{Problem, Solver};
use std::io::Write;
use std::time::Instant;

pub trait ScenarioProblem: Problem {
    fn frame_init(&mut self);
    fn initial_guess(&self) -> DVector<f64>;
    fn modify(&mut self, option: bool);
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
    file1:std::fs::File,
    file2:std::fs::File,
}

impl<P, S, LSo, LSe> Scenario<P, S, LSo, LSe>
where
    P: ScenarioProblem,
    S: Solver<P, LSo, LSe>,
    LSo: LinearSolver<MatrixType = P::HessianType>,
    LSe: LineSearch<P>,
{
    pub fn new(p: P, s: S, lso: LSo, lse: LSe) -> Self {
        let mut file1 = std::fs::File::create("new.txt").unwrap();
        file1.write_all(b"modified newton\n");
        let mut file2 = std::fs::File::create("old.txt").unwrap();
        file2.write_all(b"complete newton\n");
        Scenario {
            problem: p,
            frame: 0,
            solver: s,
            linearsolver: lso,
            ls: lse,
            file1,
            file2,
        }
    }

    pub fn step(&mut self) {
        println!("\n\nFrame: {}", self.frame);


        self.problem.frame_init();
        let initial_guess = self.problem.initial_guess();
        self.problem.modify(true);

        self.file1.write_all(self.frame.to_string().as_bytes());
        self.file1.write_all(b":\n");
        let start = Instant::now();

        let res1 = self
            .solver
            .solve(&self.problem, &self.linearsolver, &self.ls, &initial_guess,&mut self.file1);
        let duration = start.elapsed();
        println!("modifed duration: {:?}", duration);
        self.file1.write_all(b"\n");

        self.file2.write_all(self.frame.to_string().as_bytes());
        self.file2.write_all(b":\n");
        self.problem.modify(false);
        let start = Instant::now();
        let res2 = self
            .solver
            .solve(&self.problem, &self.linearsolver, &self.ls, &initial_guess,&mut self.file2);
        let duration = start.elapsed();
        println!("complete duration: {:?}", duration);
        self.file2.write_all(b"\n");
        print!("norm1: {}\n",self.problem.gradient(&res1).unwrap().norm());
        print!("norm2: {}\n",self.problem.gradient(&res2).unwrap().norm());
        print!("diff: {}\n",(&res1-&res2).amax());
        self.problem.set_all_vertices_vector(res2);


        self.frame += 1;
        self.problem.frame_end();
        #[cfg(feature = "save")]
        self.problem.save_to_file(self.frame);
    }
}
