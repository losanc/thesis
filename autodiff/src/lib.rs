mod gradient;
mod hessian;
pub use gradient::{
    constant_matrix_to_gradients, gradients_to_vector, vector_to_gradients, Gradient,
};

pub use hessian::{constant_matrix_to_hessians, hessians_to_vector, vector_to_hessians, Hessian};

pub trait MyLog: Copy {
    fn myln(self) -> Self;
}

impl MyLog for f32 {
    #[inline]
    fn myln(self) -> Self {
        self.ln()
    }
}

impl MyLog for f64 {
    #[inline]
    fn myln(self) -> Self {
        self.ln()
    }
}
