use crate::LineSearch;
mod simplels;
use crate::Problem;
use na::DVector;
use nalgebra as na;
pub use simplels::SimpleLineSearch;
pub struct NoLineSearch {}

impl<P: Problem> LineSearch<P> for NoLineSearch {
    fn search(&self, _pro: &P, _current: &DVector<f64>, direction: DVector<f64>) -> DVector<f64> {
        direction
    }
}
