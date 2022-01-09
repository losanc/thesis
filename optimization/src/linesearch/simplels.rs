/// *Simple* *L*ine *S*earch
use crate::LineSearch;
use crate::Problem;
use na::DVector;
use nalgebra as na;
pub struct SimpleLineSearch {
    pub alpha: f64,
}

impl<P: Problem> LineSearch<P> for SimpleLineSearch {
    fn search(&self, pro: &P, current: &DVector<f64>, direction: DVector<f64>) -> DVector<f64> {
        let mut scalar = 1.0;
        let check = pro.gradient(current).unwrap().dot(&direction);
        let dir: DVector<f64>;
        if check <= 0.0 {
            dir = -direction
        } else {
            dir = direction
        }
        while pro.apply(current) < pro.apply(&(current - scalar * &dir)) {
            scalar *= self.alpha;
        }
        print!("scalar {}\n", scalar);
        scalar * dir
    }
}
