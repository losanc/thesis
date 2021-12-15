mod scenarios;
use crate::scenarios::OneScenario;
use crate::scenarios::Scenario;
use optimization::linearsolver::NewtonCG;
use optimization::linearsolver::NoPre;
use optimization::linesearch::*;
use optimization::solver::NewtonSolver;
use optimization::LinearSolver;
use std::time::Instant;
fn main() {
    let problem = OneScenario::new();
    let solver = NewtonSolver {};
    let linearsolver = NewtonCG::<NoPre<_>>::new();
    // let linesearch = SimpleLineSearch { alpha: 0.9 };
    let linesearch = NoLineSearch {};
    let mut a = Scenario::new(problem, solver, linearsolver, linesearch);
    for _i in 0..100 {
        let start = Instant::now();
        a.step();
        let duration = start.elapsed();
        println!("duration: {:?}", duration);
    }
}
