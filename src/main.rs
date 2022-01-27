mod scenarios;
mod static_object;
use crate::scenarios::*;
use mesh::*;
use optimization::*;
use std::time::Instant;

// fn main() {
//     // let problem = BouncingScenario::new();
//     let problem = BouncingUpdateScenario::new();
//     let solver = NewtonSolver {};
//     // let linearsolver = NewtonCG::<NoPre<_>>::new();
//     let linearsolver = PivLU {};
//     // let linesearch = SimpleLineSearch { alpha: 0.9 };
//     let linesearch = NoLineSearch {};
//     let mut a = Scenario::new(problem, solver, linearsolver, linesearch);
//     for _i in 0..500 {
//         let start = Instant::now();
//         a.step();
//         let duration = start.elapsed();
//         // println!("duration: {:?}", duration);
//     }
// }

fn main() {
    let mesh = armadillo();
    mesh.save_to_obj("test.obj");
}
