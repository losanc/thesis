use crate::LineSearch;
mod simplels;
use crate::Problem;
use na::DVector;
use nalgebra as na;
pub use simplels::SimpleLineSearch;
pub struct NoLineSearch {}

impl<P: Problem> LineSearch<P> for NoLineSearch {
    fn search(&self, pro: &P, current: &DVector<f64>, direction: DVector<f64>) -> DVector<f64> {
        // let check = pro.gradient(current).unwrap().transpose() * &direction;
        // let check = check[(0, 0)];
        // if check <= 0.0 {
        //     -direction
        // } else {
        //     direction
        // }
        direction
    }
}
