use crate::scenarios::ScenarioProblem;
use autodiff::*;
use mesh::MeshType;
use mesh::Plane;
use na::DVector;
use nalgebra as na;
use nalgebra::SVector;
use nalgebra_sparse as nas;
use nas::CooMatrix;
use num::{One, Zero};
use optimization::Problem;
use std::string::String;

const E: f64 = 1e6;
const NU: f64 = 0.33;
const MIU: f64 = E / (2.0 * (1.0 + NU));
const LAMBDA: f64 = (E * NU) / ((1.0 + NU) * (1.0 - 2.0 * NU));
const KETA: f64 = 1e8;

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

struct Inertia {
    x_tao: DVector<f64>,
    g_vec: DVector<f64>,
    dt: f64,
    mass: CooMatrix<f64>,
}

struct Elastic {
    mesh: Plane,
}

struct Bounce {
    keta: f64,
}

impl Problem for Inertia {
    type HessianType = CooMatrix<f64>;
    fn apply(&self, x: &DVector<f64>) -> f64 {
        let temp = x - &self.x_tao - &self.g_vec * (self.dt * self.dt);
        let res = temp.dot(&(&self.mass * &temp));
        res / (2.0 * self.dt * self.dt)
    }

    fn gradient(&self, x: &DVector<f64>) -> Option<DVector<f64>> {
        let res_grad = &self.mass * (x - &self.x_tao - &self.g_vec * (self.dt * self.dt));
        Some(res_grad / (self.dt * self.dt))
    }

    fn hessian(&self, _x: &DVector<f64>) -> Option<CooMatrix<f64>> {
        Some(&self.mass / (self.dt * self.dt))
    }
}

impl Problem for Bounce {
    type HessianType = CooMatrix<f64>;
    fn apply(&self, x: &DVector<f64>) -> f64 {
        let mut res = 0.0;
        x.iter().skip(1).step_by(2).for_each(|x_i| {
            if *x_i < 0.0 {
                res -= self.keta * x_i * x_i * x_i;
            }
        });
        res
    }

    fn gradient(&self, x: &DVector<f64>) -> Option<DVector<f64>> {
        let mut res = DVector::zeros(x.len());
        x.iter()
            .zip(res.iter_mut())
            .skip(1)
            .step_by(2)
            .for_each(|(x_i, r_i)| {
                if *x_i < 0.0 {
                    *r_i -= 3.0 * self.keta * x_i * x_i;
                }
            });
        Some(res)
    }
    fn hessian(&self, x: &DVector<f64>) -> Option<CooMatrix<f64>> {
        let mut res = CooMatrix::new(x.len(), x.len());
        x.iter()
            .zip(0..x.len())
            .skip(1)
            .step_by(2)
            .for_each(|(x_i, i_i)| {
                if *x_i < 0.0 {
                    res.push(i_i, i_i, -6.0 * self.keta * x_i);
                }
            });
        Some(res)
    }
}

impl Problem for Elastic {
    type HessianType = CooMatrix<f64>;
    fn apply(&self, x: &DVector<f64>) -> f64 {
        let mut vert_vec = SVector::<f64, 6>::zeros();
        let mut res = 0.0;
        for i in 0..self.mesh.n_pris() {
            let indices = self.mesh.primitive_to_ind_vector(i);
            vert_vec
                .iter_mut()
                .zip(indices.iter())
                .for_each(|(g_i, i)| *g_i = x[*i]);

            let inv_mat = self.mesh.m_inv(i);
            let square = self.mesh.volume(i);
            energy_function!(vert_vec, ene, mat, inv_mat, square, f64);
            res += ene;
        }
        res
    }
    fn gradient(&self, x: &DVector<f64>) -> Option<DVector<f64>> {
        let mut vert_vec = SVector::<f64, 6>::zeros();
        let mut res = DVector::zeros(x.len());
        for i in 0..self.mesh.n_pris() {
            let indices = self.mesh.primitive_to_ind_vector(i);
            vert_vec
                .iter_mut()
                .zip(indices.iter())
                .for_each(|(g_i, i)| *g_i = x[*i]);
            let vert_gradient_vec = vector_to_gradients(&vert_vec);

            let inv_mat = self.mesh.m_inv(i);
            let inv_mat = constant_matrix_to_gradients(&inv_mat);
            let square = self.mesh.volume(i);
            energy_function!(vert_gradient_vec, ene, mat, inv_mat, square, Gradient<6>);
            let grad = ene.gradient();
            indices
                .iter()
                .zip(grad.iter())
                .for_each(|(i_i, g_i)| res[*i_i] += g_i);
        }
        Some(res)
    }

    fn hessian(&self, x: &DVector<f64>) -> Option<CooMatrix<f64>> {
        let mut vert_vec = SVector::<f64, 6>::zeros();
        let mut res = CooMatrix::<f64>::new(x.len(), x.len());
        for i in 0..self.mesh.n_pris() {
            let indices = self.mesh.primitive_to_ind_vector(i);
            vert_vec
                .iter_mut()
                .zip(indices.iter())
                .for_each(|(g_i, i)| *g_i = x[*i]);
            let vert_gradient_vec = vector_to_hessians(&vert_vec);
            let inv_mat = self.mesh.m_inv(i);
            let inv_mat = constant_matrix_to_hessians(&inv_mat);
            let square = self.mesh.volume(i);
            energy_function!(vert_gradient_vec, ene, mat, inv_mat, square, Hessian<6>);
            let small_hessian = ene.hessian();
            for i in 0..6 {
                for j in 0..6 {
                    res.push(indices[i], indices[j], small_hessian[(i, j)]);
                }
            }
        }
        Some(res)
    }
}

pub struct BouncingScenario {
    x_old: DVector<f64>,
    inertia: Inertia,
    elastic: Elastic,
    bounce: Bounce,
    plane: Plane,
    dt: f64,
    name: String,
}

impl Problem for BouncingScenario {
    type HessianType = CooMatrix<f64>;
    fn apply(&self, x: &DVector<f64>) -> f64 {
        let mut res = 0.0;
        res += self.inertia.apply(x);
        res += self.elastic.apply(x);
        res += self.bounce.apply(x);
        res
    }

    fn gradient(&self, x: &DVector<f64>) -> Option<DVector<f64>> {
        let mut res = DVector::<f64>::zeros(x.len());
        res += self.inertia.gradient(x)?;
        res += self.elastic.gradient(x)?;
        res += self.bounce.gradient(x)?;
        Some(res)
    }

    fn hessian(&self, x: &DVector<f64>) -> Option<Self::HessianType> {
        let mut res = CooMatrix::<f64>::new(x.len(), x.len());
        res = res + self.inertia.hessian(x).unwrap();
        res = res + self.elastic.hessian(x).unwrap();
        res = res + self.bounce.hessian(x).unwrap();
        Some(res)
    }
}

impl ScenarioProblem for BouncingScenario {
    fn initial_guess(&self) -> DVector<f64> {
        // TODO
        self.inertia.x_tao.clone()
    }
    fn set_all_vertices_vector(&mut self, vertices: DVector<f64>) {
        let velocity = (&vertices - &self.x_old) / self.dt;
        // velocity *= 0.995;
        self.plane.set_all_velocities_vector(velocity);
        self.plane.set_all_vertices_vector(vertices);
    }
    fn save_to_file(&self, frame: usize) {
        self.plane
            .save_to_obj(format!("output/{}{}.obj", self.name, frame));
    }
    fn frame_init(&mut self) {
        self.x_old = self.plane.all_vertices_to_vector();
        let v_old = self.plane.all_velocities_to_vector();
        self.inertia.x_tao = &self.x_old + self.dt * &v_old;
    }
}

impl BouncingScenario {
    pub fn new() -> Self {
        let r = 10;
        let c = 10;
        let mut p = Plane::new(r, c);
        let mut vec = p.all_vertices_to_vector();

        let mut g_vec = DVector::zeros(p.dim() * p.n_verts());
        for i in 0..p.n_verts() {
            g_vec[2 * i + 1] = -9.8;
            vec[2 * i + 1] += 1.0;
        }
        p.set_all_vertices_vector(vec);
        p.save_to_obj(format!("output/Bounce0.obj"));

        Self {
            dt: 0.01,
            plane: p.clone(),
            name: String::from("Bounce"),
            x_old: DVector::<f64>::zeros(1),
            inertia: Inertia {
                x_tao: DVector::<f64>::zeros(1),
                g_vec: g_vec,
                dt: 0.01,
                mass: p.mass_matrix(),
            },
            elastic: Elastic { mesh: p },
            bounce: Bounce { keta: KETA },
        }
    }
}
