// mod simple_new;
mod bouncing_update;
pub use bouncing_update::BouncingScenario as OneScenario;
use na::DVector;
use nalgebra as na;
use optimization::LineSearch;
use optimization::LinearSolver;
use optimization::{Problem, Solver};

pub trait ScenarioProblem: Problem {
    fn frame_init(&mut self);
    fn initial_guess(&self) -> DVector<f64>;
    fn set_all_vertices_vector(&mut self, vertices: DVector<f64>);
    fn save_to_file(&self, frame: usize);
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
    x: DVector<f64>,
    ls: LSe,
}

impl<P, S, LSo, LSe> Scenario<P, S, LSo, LSe>
where
    P: ScenarioProblem,
    S: Solver<P, LSo, LSe>,
    LSo: LinearSolver<MatrixType = P::HessianType>,
    LSe: LineSearch<P>,
{
    pub fn new(p: P, s: S, lso: LSo, lse: LSe) -> Self {
        Scenario {
            problem: p,
            frame: 0,
            solver: s,
            linearsolver: lso,
            x: DVector::<f64>::zeros(1),
            ls: lse,
        }
    }

    pub fn step(&mut self) {
        self.problem.frame_init();
        self.x = self.problem.initial_guess();
        let res = self
            .solver
            .solve(&self.problem, &self.linearsolver, &self.ls, &self.x);
        self.problem.set_all_vertices_vector(res);
        self.frame += 1;
        println!("Frame: {}", self.frame);
        #[cfg(feature = "save")]
        self.problem.save_to_file(self.frame);
    }
}
