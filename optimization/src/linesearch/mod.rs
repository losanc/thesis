use crate::LineSearch;
// mod simplels;
// pub use simplels::SimpleLineSearch;
use nalgebra as na;
use na::DVector;
use crate::Problem;
pub struct NoLineSearch {}

impl<P: Problem> LineSearch<P> for NoLineSearch {
    fn search(&self, pro: &P, current: &DVector<f64>, direction: DVector<f64>) -> DVector<f64> {
        let check = pro.gradient(current).unwrap().transpose() * &direction;
        let check = check[(0, 0)];
        if check <= 0.0 {
            direction
        } else {
            -direction
        }
    }
}
