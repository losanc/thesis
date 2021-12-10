mod scenarios;
use crate::scenarios::Scenario;
use crate::scenarios::SimpleScenario;
use optimization::linearsolver::NewtonCG;
use optimization::linearsolver::NoPre;
use optimization::linesearch::NoLineSearch;
use optimization::solver::NewtonSolver;
use optimization::LinearSolver;
use std::time::{Duration, Instant};
fn main() {
    let problem = SimpleScenario::new();
    let solver = NewtonSolver {};
    let linearsolver = NewtonCG::<NoPre<_>>::new();
    let linesearch = NoLineSearch {};
    let mut a = Scenario::new(problem, solver, linearsolver, linesearch);
    for i in 0..1000 {
        let start = Instant::now();
        a.step();
        let duration = start.elapsed();
    println!("duration: {:?}", duration);
    }
}
