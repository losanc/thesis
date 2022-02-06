use crate::scenarios::ScenarioProblem;
use autodiff::*;
use mesh::MeshType;
use mesh::Plane;
use na::DVector;
use nalgebra as na;
use nalgebra::SVector;
use nalgebra_sparse as nas;
use nas::{CooMatrix, CsrMatrix};
use num::{One, Zero};
use optimization::Problem;
use std::string::String;

const E: f64 = 1e4;
const NU: f64 = 0.33;
const MIU: f64 = E / (2.0 * (1.0 + NU));
const LAMBDA: f64 = (E * NU) / ((1.0 + NU) * (1.0 - 2.0 * NU));

macro_rules! energy_function {
    ($vec:ident, $ene:ident,$mat:ident,$inv_mat:ident, $square:expr, $type:ty) => {
        let $mat = na::matrix![
                $vec[4]-$vec[0], $vec[2]-$vec[0];
                $vec[5]-$vec[1], $vec[3]-$vec[1];
            ];
        let $mat = $mat*$inv_mat;
        let $mat  = ($mat.transpose() * $mat  -
        na::matrix![
            <$type>::one(), <$type>::zero();
            <$type>::zero(), <$type>::one();
        ]) *(<$type>::one()*0.5);

        let $ene = ($mat.transpose()*$mat).trace()*(<$type>::one()*MIU) +
         $mat.trace()*$mat.trace()*(<$type>::one()*(0.5*LAMBDA));
         let $ene = $ene *(<$type>::one()*$square);

    };
}

pub struct SimpleScenario {
    x_tao: DVector<f64>,
    x_old: DVector<f64>,
    mass: CooMatrix<f64>,
    dt: f64,
    p: Plane,
    name: String,
    dim: usize,
}

impl Problem for SimpleScenario {
    type HessianType = CooMatrix<f64>;
    fn apply(&self, x: &DVector<f64>) -> f64 {
        let res = (x - &self.x_tao).transpose() * (&self.mass * (x - &self.x_tao));
        res[(0, 0)]
    }
    fn gradient(&self, x: &DVector<f64>) -> Option<DVector<f64>> {
        let mut res_grad = &self.mass * (x - &self.x_tao) / (self.dt * self.dt);
        // TODO: because only svector used inside, so it has to be fixed sive when compiling time.
        let mut vert_vec = SVector::<f64, 6>::zeros();
        for i in 0..self.p.n_pris() {
            let indices = self.p.primitive_to_ind_vector(i);
            vert_vec
                .iter_mut()
                .zip(indices.iter())
                .for_each(|(g_i, i)| *g_i = x[*i]);
            let vert_gradient_vec = vector_to_gradients(&vert_vec);

            let inv_mat = self.p.m_inv(i);
            let inv_mat = constant_matrix_to_gradients(&inv_mat);

            energy_function!(
                vert_gradient_vec,
                ene,
                mat,
                inv_mat,
                // todo
                // self.square[index],
                0.5,
                Gradient<6>
            );
            let grad = ene.gradient();
            indices
                .iter()
                .zip(grad.iter())
                .for_each(|(i_i, g_i)| res_grad[*i_i] += g_i);
        }

        // for i in 0..self.p.n_fixed_verts() {
        //     for j in 0..self.dim {
        //         res_grad[i * self.dim + j] = 0.0;
        //     }
        // }
        Some(res_grad)
    }

    fn hessian(&self, x: &DVector<f64>) -> Option<Self::HessianType> {
        let res = self.mass.clone();
        let mut res = res / (self.dt * self.dt);
        let mut vert_vec = SVector::<f64, 6>::zeros();
        for i in 0..self.p.n_pris() {
            let indices = self.p.primitive_to_ind_vector(i);
            vert_vec
                .iter_mut()
                .zip(indices.iter())
                .for_each(|(g_i, i)| *g_i = x[*i]);
            let vert_gradient_vec = vector_to_hessians(&vert_vec);
            let inv_mat = self.p.m_inv(i);
            let inv_mat = constant_matrix_to_hessians(&inv_mat);

            energy_function!(
                vert_gradient_vec,
                ene,
                mat,
                inv_mat,
                // todo
                // self.square[index],
                0.5,
                Hessian<6>
            );
            let small_hessian = ene.hessian();
            // print!("{}\n",small_hessian);
            for i in 0..6 {
                for j in 0..6 {
                    res.push(indices[i], indices[j], small_hessian[(i, j)]);
                }
            }
        }

        Some(res)
    }
}

impl ScenarioProblem for SimpleScenario {
    fn initial_guess(&self) -> DVector<f64> {
        // TODO
        self.x_tao.clone()
    }
    fn set_all_vertices_vector(&mut self, vertices: &DVector<f64>) {
        let mut velocity = (vertices - &self.x_old) / self.dt;
        // velocity *= 0.995;
        self.p.set_all_velocities_vector(&velocity);
        self.p.set_all_vertices_vector(vertices);
    }
    fn save_to_file(&self, frame: usize) {
        self.p.save_to_obj(format!("{}{}.obj", self.name, frame));
    }
    fn frame_init(&mut self) {
        self.x_old = self.p.all_vertices_to_vector();
        let old_v = self.p.all_velocities_to_vector();
        self.x_tao = &self.x_old + self.dt * &old_v;
    }
}

impl SimpleScenario {
    pub fn new() -> Self {
        let r = 10;
        let c = 10;
        let mut p = Plane::new(r, c);
        let x_tao = DVector::zeros(p.dim() * r * c);
        let mut vec = p.all_vertices_to_vector();
        vec[0] = -0.5;
        vec[1] = -0.5;
        p.set_all_vertices_vector(&vec);
        p.save_to_obj::<_>("test.obj");
        Self {
            dt: 0.01,
            x_tao: x_tao,
            mass: p.mass_matrix(),
            x_old: vec,
            dim: p.dim(),
            p: p,
            name: String::from("Simple"),
        }
    }
}
