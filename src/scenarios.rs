use na::DVector;
use nalgebra as na;
use optimization::run_when_logging;
use optimization::LineSearch;
use optimization::LinearSolver;
use optimization::{Problem, Solver};
#[allow(unused_imports)]
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
    buf_writter: std::io::BufWriter<std::fs::File>,
}

impl<P, S, LSo, LSe> Scenario<P, S, LSo, LSe>
where
    P: ScenarioProblem,
    S: Solver<P, LSo, LSe>,
    LSo: LinearSolver<MatrixType = P::HessianType>,
    LSe: LineSearch<P>,
{
    pub fn new(
        p: P,
        s: S,
        lso: LSo,
        lse: LSe,
        #[cfg(feature = "log")] dirname: String,
        #[cfg(feature = "log")] filename: String,
        #[cfg(feature = "log")] comment: String,
    ) -> Self {
        #[cfg(feature = "log")]
        let file;
        #[cfg(feature = "log")]
        let mut buf_writter;

        run_when_logging!(
            std::fs::create_dir_all(&dirname).unwrap();
            file = std::fs::File::create(dirname+"/"+&filename).unwrap();
            buf_writter = std::io::BufWriter::new(file);
            writeln!(buf_writter, "{}", comment).unwrap();
        );

        Scenario {
            problem: p,
            frame: 0,
            solver: s,
            linearsolver: lso,
            ls: lse,
            #[cfg(feature = "log")]
            buf_writter,
        }
    }

    pub fn step(&mut self) {
        self.problem.frame_init();
        let initial_guess = self.problem.initial_guess();
        run_when_logging!(writeln!(self.buf_writter, "\nFrame: {}", self.frame).unwrap());
        let res2 = self.solver.solve::<std::io::BufWriter<std::fs::File>>(
            &mut self.problem,
            &self.linearsolver,
            &self.ls,
            &initial_guess,
            #[cfg(feature = "log")]
            &mut self.buf_writter,
        );

        #[cfg(feature = "log")]
        {
            self.buf_writter.flush().unwrap();
        }

        self.problem.set_all_vertices_vector(res2);
        self.frame += 1;
        self.problem.frame_end();
        #[cfg(feature = "save")]
        self.problem.save_to_file(self.frame);
    }
}
