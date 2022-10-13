/// *Simple* *L*ine *S*earch
use crate::LineSearch;
use crate::Problem;
use na::DVector;
use nalgebra as na;
pub struct SimpleLineSearch {
    pub alpha: f64,
    pub tol: f64,
    pub epi: f64,
}

impl<P: Problem> LineSearch<P> for SimpleLineSearch {
    fn search(&self, pro: &P, current: &DVector<f64>, direction: &DVector<f64>) -> f64 {
        let mut scalar = 1.0;
        let check = pro.gradient(current).unwrap().dot(&direction);
        if check <= 0.0 {
            scalar = -1.0
        }
        while pro.apply(current) + 1e-4*scalar*check < pro.apply(&(current - scalar * direction)) {
            scalar *= self.alpha;
            if scalar.abs() < self.tol {
                break;
            }
        }
        scalar
    }
}
