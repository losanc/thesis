pub struct SimpleLineSearch {
    alpha: f64,
}

impl<P: Problem> LineSearch for SimpleLineSearch<P> {
    fn search(&self, pro: &P, current: &DVector<f64>, direction: DVector<f64>) -> DVector<f64> {
        let mut scalar = 1.0;
        while pro.apply(current) >= pro.apply(current + scalar * direction) {
            scalar *= self.alpha;
        }
        current + scalar * direction
    }
}
