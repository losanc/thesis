use autodiff::*;
use nalgebra::{DMatrix, DVector, DVectorSlice, SVector};
pub trait static_object<'a> {
    /// returns true when collision happens
    fn detect(&self, co: impl Into<DVectorSlice<'a, f64>>) -> bool;
    fn energy(&self, co: impl Into<DVectorSlice<'a, f64>>) -> f64;
    fn gradient(&self, co: impl Into<DVectorSlice<'a, f64>>) -> DVector<f64>;
    fn hessian(&self, co: impl Into<DVectorSlice<'a, f64>>) -> DMatrix<f64>;
}

pub struct Ground {
    pub keta: f64,
    pub height: f64,
}

impl<'a> static_object<'a> for Ground {
    fn detect(&self, co: impl Into<DVectorSlice<'a, f64>>) -> bool {
        let co = co.into();
        co[1] <= self.height
    }
    fn energy(&self, co: impl Into<DVectorSlice<'a, f64>>) -> f64 {
        let co = co.into();
        let c = co[1];
        let d = c - self.height;
        if d < 0.0 {
            return -self.keta * d * d * d;
        }
        0.0
    }
    fn gradient(&self, co: impl Into<DVectorSlice<'a, f64>>) -> DVector<f64> {
        let co = co.into();
        let mut res = DVector::<f64>::zeros(co.len());
        let c = co[1];
        let d = c - self.height;
        if d < 0.0 {
            res[1] = -3.0 * self.keta * d * d;
        }
        res
    }
    fn hessian(&self, co: impl Into<DVectorSlice<'a, f64>>) -> DMatrix<f64> {
        let co = co.into();
        let mut res = DMatrix::<f64>::zeros(co.len(), co.len());
        let c = co[1];
        let d = c - self.height;
        if d < 0.0 {
            res[(1, 1)] = -6.0 * self.keta * d;
        }
        res
    }
}

pub struct StaticCircle {
    pub center: DVector<f64>,
    pub radius: f64,
    pub keta: f64,
}

impl<'a> static_object<'a> for StaticCircle {
    fn detect(&self, co: impl Into<DVectorSlice<'a, f64>>) -> bool {
        let co = co.into();
        (co - &self.center).norm() < self.radius
    }
    fn energy(&self, co: impl Into<DVectorSlice<'a, f64>>) -> f64 {
        let co = co.into();
        let norm = (co - &self.center).norm();
        let d = norm * norm - self.radius * self.radius;
        if d > 0.0 {
            return 0.0;
        } else {
            return -self.keta * d * d * d;
        }
    }
    fn gradient(&self, co: impl Into<DVectorSlice<'a, f64>>) -> DVector<f64> {
        let co = co.into();
        let mut res = DVector::<f64>::zeros(co.len());
        let mut gra_co = SVector::<f64, 3>::zeros();
        let mut gra_ce = SVector::<f64, 3>::zeros();

        let l = co.len();

        // make it work even co is 2d
        let mut t = gra_co.index_mut((..l, 0));
        //  or could be this
        // gra_co.index_mut((..l,0));
        t += co;

        let mut t = gra_ce.index_mut((..l, 0));

        t += &self.center;

        let gra_co = vector_to_gradients(gra_co);
        let gra_ce = constant_matrix_to_gradients(gra_ce);
        let norm = (gra_co - gra_ce).transpose() * (gra_co - gra_ce);
        let norm = norm.trace();
        let d = norm - self.radius * self.radius;
        if d.value() >= 0.0 {
            return res;
        } else {
            let r = d * d * d * -self.keta;
            res += r.gradient().index((..l, 0));
            return res;
        }
    }
    fn hessian(&self, co: impl Into<DVectorSlice<'a, f64>>) -> DMatrix<f64> {
        let co = co.into();
        let mut res = DMatrix::<f64>::zeros(co.len(), co.len());
        let mut gra_co = SVector::<f64, 3>::zeros();
        let mut gra_ce = SVector::<f64, 3>::zeros();

        let l = co.len();

        // make it work even co is 2d
        let mut t = gra_co.index_mut((..l, 0));
        //  or could be this
        // gra_co.index_mut((..l,0));
        t += co;

        let mut t = gra_ce.index_mut((..l, 0));

        t += &self.center;

        let gra_co = vector_to_hessians(gra_co);
        let gra_ce = constant_matrix_to_hessians(gra_ce);
        let norm = (gra_co - gra_ce).transpose() * (gra_co - gra_ce);
        let norm = norm.trace();
        let d = norm - self.radius * self.radius;
        if d.value() >= 0.0 {
            return res;
        } else {
            let r = d * d * d * -self.keta;
            res += r.hessian().index((..l, ..l));
            return res;
        }
    }
}
