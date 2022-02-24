use crate::scenarios::ScenarioProblem;
use crate::static_object::*;
use autodiff::*;
use mesh::*;
use na::{dvector, DVector};
use nalgebra as na;
use nalgebra::SVector;
use nalgebra::{DMatrix, SMatrix};
use num::{One, Zero};
use optimization::Problem;
use std::cell::RefCell;
use std::collections::HashSet;
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
    mass: DMatrix<f64>,
}

struct Elastic {
    n_prims: usize,
    prim_connected_vert_indices: Vec<[usize; 3]>,
    vert_connected_prim_indices: Vec<Vec<usize>>,
    volumes: Vec<f64>,
    ma_invs: Vec<SMatrix<f64, 2, 2>>,
    update_list: HashSet<usize>,
    old_hessian_list: RefCell<Vec<SMatrix<f64, 6, 6>>>,
}

impl Problem for Inertia {
    type HessianType = DMatrix<f64>;
    fn apply(&self, x: &DVector<f64>) -> f64 {
        let temp = x - &self.x_tao - &self.g_vec * (self.dt * self.dt);
        let res = temp.dot(&(&self.mass * &temp));
        res / (2.0 * self.dt * self.dt)
    }

    fn gradient(&self, x: &DVector<f64>) -> Option<DVector<f64>> {
        let res_grad = &self.mass * (x - &self.x_tao - &self.g_vec * (self.dt * self.dt));
        Some(res_grad / (self.dt * self.dt))
    }

    fn hessian(&self, _x: &DVector<f64>) -> Option<Self::HessianType> {
        Some(&self.mass / (self.dt * self.dt))
    }
}

impl Problem for Elastic {
    type HessianType = DMatrix<f64>;
    fn apply(&self, x: &DVector<f64>) -> f64 {
        let mut vert_vec = SVector::<f64, 6>::zeros();
        let mut res = 0.0;
        for i in 0..self.n_prims {
            let ind = self.prim_connected_vert_indices[i];
            let indices = vec![
                ind[0] * 2,
                ind[0] * 2 + 1,
                ind[1] * 2,
                ind[1] * 2 + 1,
                ind[2] * 2,
                ind[2] * 2 + 1,
            ];
            vert_vec
                .iter_mut()
                .zip(indices.iter())
                .for_each(|(g_i, i)| *g_i = x[*i]);

            let inv_mat = self.ma_invs[i];
            let square = self.volumes[i];
            energy_function!(vert_vec, ene, mat, inv_mat, square, f64);
            res += ene;
        }
        res
    }
    fn gradient(&self, x: &DVector<f64>) -> Option<DVector<f64>> {
        let mut vert_vec = SVector::<f64, 6>::zeros();
        let mut res = DVector::zeros(x.len());
        for i in 0..self.n_prims {
            let ind = self.prim_connected_vert_indices[i];
            let indices = vec![
                ind[0] * 2,
                ind[0] * 2 + 1,
                ind[1] * 2,
                ind[1] * 2 + 1,
                ind[2] * 2,
                ind[2] * 2 + 1,
            ];

            vert_vec
                .iter_mut()
                .zip(indices.iter())
                .for_each(|(g_i, i)| *g_i = x[*i]);
            let vert_gradient_vec = vector_to_gradients(vert_vec);

            let inv_mat = self.ma_invs[i];
            let inv_mat = constant_matrix_to_gradients(inv_mat);
            let square = self.volumes[i];
            energy_function!(vert_gradient_vec, ene, mat, inv_mat, square, Gradient<6>);
            let grad = ene.gradient();
            indices
                .iter()
                .zip(grad.iter())
                .for_each(|(i_i, g_i)| res[*i_i] += g_i);
        }
        Some(res)
    }

    fn hessian(&self, x: &DVector<f64>) -> Option<Self::HessianType> {
        let mut res = DMatrix::<f64>::zeros(x.len(), x.len());

        // extract vertex list need to be updated
        // vertex id =  gradient_vec_coordiante / 2
        let update_list = self
            .update_list
            .iter()
            .map(|x| x / 2)
            .collect::<HashSet<usize>>();
        println!("update_list: {:?}", update_list.len());
        // find out all connected triangles
        let update_triangle_list = update_list
            .iter()
            .map(|x| self.vert_connected_prim_indices[*x].clone())
            .flatten()
            .collect::<HashSet<usize>>();

        let mut vert_vec = SVector::<f64, 6>::zeros();

        println!("{:?}", update_triangle_list.len());

        // now if tr
        for i in 0..self.n_prims {
            let small_hessian;
            let ind = self.prim_connected_vert_indices[i];
            let indices = vec![
                ind[0] * 2,
                ind[0] * 2 + 1,
                ind[1] * 2,
                ind[1] * 2 + 1,
                ind[2] * 2,
                ind[2] * 2 + 1,
            ];
            if update_triangle_list.contains(&i) {
                vert_vec
                    .iter_mut()
                    .zip(indices.iter())
                    .for_each(|(g_i, i)| *g_i = x[*i]);
                let vert_gradient_vec = vector_to_hessians(vert_vec);
                let inv_mat = self.ma_invs[i];
                let inv_mat = constant_matrix_to_hessians(inv_mat);
                let square = self.volumes[i];
                energy_function!(vert_gradient_vec, ene, mat, inv_mat, square, Hessian<6>);
                small_hessian = ene.hessian();
                self.old_hessian_list.borrow_mut()[i] = small_hessian.clone();
            } else {
                small_hessian = self.old_hessian_list.borrow()[i];
            }
            for i in 0..6 {
                for j in 0..6 {
                    res[(indices[i], indices[j])] += small_hessian[(i, j)];
                }
            }
        }

        Some(res)
    }
}

pub struct BouncingUpdateScenario {
    inertia: Inertia,
    elastic: RefCell<Elastic>,
    plane: Mesh2d,
    dt: f64,
    name: String,
    tol: f64,
    update_list: RefCell<HashSet<usize>>,
    ground: Ground,
    circle: StaticCircle,
    circle2: StaticCircle,
}

impl Problem for BouncingUpdateScenario {
    type HessianType = DMatrix<f64>;
    fn apply(&self, x: &DVector<f64>) -> f64 {
        let mut res = 0.0;
        res += self.inertia.apply(x);
        let elastic = self.elastic.try_borrow().unwrap();
        res += elastic.apply(x);

        for i in 0..self.plane.n_verts {
            res += self.circle.energy(x.index((2 * i..2 * i + 2, 0)));
        }
        for i in 0..self.plane.n_verts {
            res += self.circle2.energy(x.index((2 * i..2 * i + 2, 0)));
        }

        res
    }

    fn gradient(&self, x: &DVector<f64>) -> Option<DVector<f64>> {
        let mut res = DVector::<f64>::zeros(x.len());
        res += self.inertia.gradient(x)?;

        let elastic = self.elastic.try_borrow().unwrap();
        res += elastic.gradient(x)?;

        for i in 0..self.plane.n_verts {
            let mut slice = res.index_mut((2 * i..2 * i + 2, 0));
            slice += self.circle.gradient(x.index((2 * i..2 * i + 2, 0)));
        }
        for i in 0..self.plane.n_verts {
            let mut slice = res.index_mut((2 * i..2 * i + 2, 0));
            slice += self.circle2.gradient(x.index((2 * i..2 * i + 2, 0)));
        }

        let mut update_list = self.update_list.try_borrow_mut().unwrap();
        update_list.clear();
        for (i, r) in res.iter().enumerate() {
            if r.abs() > self.tol {
                update_list.insert(i);
            }
        }
        Some(res)
    }

    fn hessian(&self, x: &DVector<f64>) -> Option<Self::HessianType> {
        let mut res = DMatrix::<f64>::zeros(x.len(), x.len());
        res = res + self.inertia.hessian(x)?;

        let mut elastic = self.elastic.try_borrow_mut().unwrap();
        elastic.update_list = self.update_list.try_borrow().unwrap().clone();
        res = res + elastic.hessian(x)?;

        for i in 0..self.plane.n_verts {
            let mut slice = res.index_mut((2 * i..2 * i + 2, 2 * i..2 * i + 2));
            slice += self.circle.hessian(x.index((2 * i..2 * i + 2, 0)));
        }
        for i in 0..self.plane.n_verts {
            let mut slice = res.index_mut((2 * i..2 * i + 2, 2 * i..2 * i + 2));
            slice += self.circle2.hessian(x.index((2 * i..2 * i + 2, 0)));
        }

        Some(res)
    }
}

impl ScenarioProblem for BouncingUpdateScenario {
    fn initial_guess(&self) -> DVector<f64> {
        // TODO
        self.inertia.x_tao.clone()
    }
    fn set_all_vertices_vector(&mut self, vertices: DVector<f64>) {
        let velocity = (&vertices - &self.plane.verts) / self.dt;
        self.plane.velos = velocity;
        self.plane.verts = vertices;
    }
    fn save_to_file(&self, frame: usize) {
        self.plane
            .save_to_obj(format!("output/{}{}.obj", self.name, frame));
    }
    fn frame_init(&mut self) {
        self.inertia.x_tao = &self.plane.verts + self.dt * &self.plane.velos;
    }
    fn frame_end(&mut self) {}
}

impl BouncingUpdateScenario {
    pub fn new() -> Self {
        println!("new");
        let mut p = circle(1.0, 5, None);
        // let mut p = plane(r, c, None);
        let vec = &mut p.verts;

        let mut g_vec = DVector::zeros(2 * p.n_verts);
        for i in 0..p.n_verts {
            g_vec[2 * i + 1] = -9.8;
            vec[2 * i + 1] += 3.0;
        }
        #[cfg(feature = "save")]
        p.save_to_obj(format!("output/bouncingupdate0.obj"));

        Self {
            dt: 0.01,

            name: String::from("bouncingupdate"),
            inertia: Inertia {
                x_tao: DVector::<f64>::zeros(1),
                g_vec,
                dt: 0.01,
                mass: DMatrix::from_diagonal(&p.masss),
            },

            elastic: RefCell::<_>::new(Elastic {
                n_prims: p.n_prims,
                prim_connected_vert_indices: p.prim_connected_vert_indices.clone(),
                vert_connected_prim_indices: p.vert_connected_prim_indices.clone(),
                volumes: p.volumes.clone(),
                ma_invs: p.ma_invs.clone(),
                update_list: HashSet::<usize>::new(),
                old_hessian_list: RefCell::<_>::new(vec![SMatrix::<f64, 6, 6>::zeros(); p.n_prims]),
            }),
            plane: p,
            update_list: RefCell::<_>::new(HashSet::<usize>::new()),
            tol: 10.0,
            ground: Ground {
                keta: KETA,
                height: -1.0,
            },
            circle: StaticCircle {
                keta: KETA,
                radius: 1.0,
                center: dvector![-1.5, 0.0],
            },
            circle2: StaticCircle {
                keta: KETA,
                radius: 1.0,
                center: dvector![1.5, 0.0],
            },
        }
    }
}