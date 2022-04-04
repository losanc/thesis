use autodiff::*;
use mesh::*;
use na::{dvector, DVector};
use nalgebra as na;
use nalgebra::SVector;
use nalgebra::{DMatrix, SMatrix};
use optimization::*;
use std::cell::RefCell;
use std::collections::HashSet;
use std::convert::TryInto;
use std::string::String;
use thesis::my_newton::MyProblem;

use thesis::scenarios::Scenario;
use thesis::scenarios::ScenarioProblem;
use thesis::static_object::*;

const E: f64 = 1e6;
const NU: f64 = 0.33;
const MIU: f64 = E / (2.0 * (1.0 + NU));
const LAMBDA: f64 = (E * NU) / ((1.0 + NU) * (1.0 - 2.0 * NU));
const KETA: f64 = 1e8;
const DT: f64 = 0.03;

const DIM: usize = 2;
const CO_NUM: usize = DIM * (DIM + 1);
const DAMP: f64 = 1.0;

type EnergyType = NeoHookean2d;

pub struct BouncingUpdateScenario {
    plane: Mesh2d,
    dt: f64,
    name: String,
    energy: EnergyType,
    circle: StaticCircle,
    circle2: StaticCircle,
    old_hessian_list: RefCell<Vec<SMatrix<f64, CO_NUM, CO_NUM>>>,
    x_tao: DVector<f64>,
    g_vec: DVector<f64>,
    mass: DMatrix<f64>,
}

impl BouncingUpdateScenario {
    fn elastic_my_hessian(
        &self,
        x: &DVector<f64>,
        active_set: &std::collections::HashSet<usize>,
    ) -> DMatrix<f64> {
        let mut res = DMatrix::zeros(x.len(), x.len());

        let update_triangle_list = active_set
            .iter()
            .map(|x| self.plane.vert_connected_prim_indices[x / DIM].clone())
            .flatten()
            .collect::<HashSet<usize>>();

        for i in 0..self.plane.n_prims {
            let small_hessian;
            let indices = self.plane.get_indices(i);
            if update_triangle_list.contains(&i) {
                let mut vert_vec = SVector::<f64, CO_NUM>::zeros();
                vert_vec
                    .iter_mut()
                    .zip(indices.iter())
                    .for_each(|(g_i, i)| *g_i = x[*i]);
                let energy: Hessian<CO_NUM> = self.plane.prim_energy(i, &self.energy, vert_vec);
                small_hessian = energy.hessian();
                self.old_hessian_list.borrow_mut()[i] = small_hessian;
            } else {
                small_hessian = self.old_hessian_list.borrow()[i];
            }
            for i in 0..CO_NUM {
                for j in 0..CO_NUM {
                    res[(indices[i], indices[j])] += small_hessian[(i, j)];
                }
            }
        }
        res
    }

    fn inertia_apply(&self, x: &DVector<f64>) -> f64 {
        let temp = x - &self.x_tao - &self.g_vec * (self.dt * self.dt);
        let res = temp.dot(&(&self.mass * &temp));
        res / (2.0 * self.dt * self.dt)
    }
    fn inertia_gradient(&self, x: &DVector<f64>) -> DVector<f64> {
        let res_grad = &self.mass * (x - &self.x_tao - &self.g_vec * (self.dt * self.dt));
        res_grad / (self.dt * self.dt)
    }
    fn inertia_hessian(&self, _x: &DVector<f64>) -> DMatrix<f64> {
        self.mass.clone() / (self.dt * self.dt)
    }

    fn collision_apply(&self, x: &DVector<f64>) -> f64 {
        let mut res = 0.0;
        for i in 0..self.plane.n_verts {
            res += self.circle.energy(x.index((2 * i..2 * i + 2, 0)));
            res += self.circle2.energy(x.index((2 * i..2 * i + 2, 0)));
        }
        res
    }

    fn collision_gradient(&self, x: &DVector<f64>) -> DVector<f64> {
        let mut res = DVector::zeros(x.len());
        for i in 0..self.plane.n_verts {
            let mut slice = res.index_mut((2 * i..2 * i + 2, 0));
            slice += self.circle.gradient(x.index((2 * i..2 * i + 2, 0)));
        }
        for i in 0..self.plane.n_verts {
            let mut slice = res.index_mut((2 * i..2 * i + 2, 0));
            slice += self.circle2.gradient(x.index((2 * i..2 * i + 2, 0)));
        }
        res
    }

    fn collision_hessian(&self, x: &DVector<f64>) -> DMatrix<f64> {
        let mut res = DMatrix::zeros(x.len(), x.len());
        for i in 0..self.plane.n_verts {
            let mut slice = res.index_mut((2 * i..2 * i + 2, 2 * i..2 * i + 2));
            slice += self.circle.hessian(x.index((2 * i..2 * i + 2, 0)));
        }
        for i in 0..self.plane.n_verts {
            let mut slice = res.index_mut((2 * i..2 * i + 2, 2 * i..2 * i + 2));
            slice += self.circle2.hessian(x.index((2 * i..2 * i + 2, 0)));
        }
        res
    }
}

static mut count: usize = 0;

impl Problem for BouncingUpdateScenario {
    type HessianType = DMatrix<f64>;
    fn apply(&self, x: &DVector<f64>) -> f64 {
        let mut res = 0.0;
        res += self.inertia_apply(x);
        res += self.plane.elastic_apply(x, &self.energy);
        res += self.collision_apply(x);

        res
    }

    fn gradient(&self, x: &DVector<f64>) -> Option<DVector<f64>> {
        let mut res = DVector::<f64>::zeros(x.len());
        res += self.inertia_gradient(x);
        res += self.plane.elastic_gradient(x, &self.energy);
        res += self.collision_gradient(x);
        Some(res)
    }

    fn hessian(&self, x: &DVector<f64>) -> Option<Self::HessianType> {
        let mut res = DMatrix::<f64>::zeros(x.len(), x.len());
        res += self.inertia_hessian(x);
        res += self.plane.elastic_hessian(x, &self.energy);
        

        // let surface_vertices = self
        //     .plane
        //     .surface
        //     .as_ref()
        //     .unwrap()
        //     .iter()
        //     .flatten()
        //     .collect::<std::collections::HashSet<_>>();
        // let zero = DVector::zeros(x.len());
        // let zero_row = zero.transpose();
        // for v_i in surface_vertices {
        //     elastic_matrix.set_column(v_i * 2, &zero);
        //     elastic_matrix.set_column(v_i * 2 + 1, &zero);
        //     elastic_matrix.set_row(v_i * 2, &zero_row);
        //     elastic_matrix.set_row(v_i * 2 + 1, &zero_row);
        //     elastic_matrix[(2 * v_i, 2 * v_i)] = 1.0;
        //     elastic_matrix[(2 * v_i + 1, 2 * v_i + 1)] = 1.0;
        // }
        // // invere.as_mut_slice().iter_mut().for_each(|x| *x = *x*1e7);
        // let inverser = elastic_matrix.clone().try_inverse().unwrap();

        // // let ident = DMatrix::identity(x.len(), x.len());
        // // let inverse_matrix = elastic_matrix.clone().svd(true,true).solve(&ident,1e-10).unwrap();
        unsafe {
            let elastic_matrix = self.plane.elastic_hessian(x, &self.energy);
            let data = elastic_matrix
                .as_slice()
                .iter()
                .map(|x| x.to_ne_bytes())
                .flatten()
                .collect::<Vec<u8>>();

            let mut file =
                std::fs::File::create(format!("output/matrix/hessian{count}.txt")).unwrap();

            use std::io::Write;
            file.write_all(&data).unwrap();
            self.plane.save_to_obj(format!("output/matrix/mesh{count}.obj"));
            count += 1;
        }

        // let data = inverse_matrix
        //     .as_slice()
        //     .iter()
        //     .map(|x| x.to_ne_bytes())
        //     .flatten()
        //     .collect::<Vec<u8>>();
        // unsafe {
        //     let mut file =
        //         std::fs::File::create(format!("output/matrix/hessian{count}.txt")).unwrap();

        //     count += 1;

        //     use std::io::Write;
        //     file.write_all(&data).unwrap();
        // }
        res += self.collision_hessian(x);
        Some(res)
    }
}

impl MyProblem for BouncingUpdateScenario {
    fn my_gradient(
        &self,
        x: &DVector<f64>,
    ) -> (
        Option<DVector<f64>>,
        Option<std::collections::HashSet<usize>>,
    ) {
        let res = self.gradient(x).unwrap();
        let mut active_set = std::collections::HashSet::<usize>::new();
        let max = 0.6 * res.amax();
        for (i, r) in res.iter().enumerate() {
            if r.abs() > max {
                active_set.insert(i);
            }
        }
        (Some(res), Some(active_set))
    }

    fn my_hessian(
        &self,
        x: &DVector<f64>,
        active_set: &std::collections::HashSet<usize>,
    ) -> Option<<Self as Problem>::HessianType> {
        let mut res = DMatrix::<f64>::zeros(x.len(), x.len());
        res = res + self.inertia_hessian(x);
        res = res + self.elastic_my_hessian(x, active_set);
        res = res + self.collision_hessian(x);
        Some(res)
    }
}

impl ScenarioProblem for BouncingUpdateScenario {
    fn initial_guess(&self) -> DVector<f64> {
        self.plane.verts.clone()
    }
    fn set_all_vertices_vector(&mut self, vertices: DVector<f64>) {
        let velocity = DAMP * ((&vertices - &self.plane.verts) / self.dt);
        self.plane.velos = velocity;
        self.plane.verts = vertices;
    }
    fn save_to_file(&self, frame: usize) {
        self.plane
            .save_to_obj(format!("output/{}{}.obj", self.name, frame));
    }
    fn frame_init(&mut self) {
        self.x_tao = &self.plane.verts + self.dt * &self.plane.velos;
    }
    fn frame_end(&mut self) {}
}

impl BouncingUpdateScenario {
    pub fn new(name: &str) -> Self {
        let mut p = plane(2, 2, None);
        p.save_to_obj("a.obj");
        let vec = &mut p.verts;

        let mut g_vec = DVector::zeros(2 * p.n_verts);
        for i in 0..p.n_verts {
            g_vec[2 * i + 1] = -9.8;
            vec[2 * i + 1] += 10.0;
        }

        Self {
            dt: DT,
            mass: DMatrix::from_diagonal(&p.masss),
            energy: EnergyType {
                mu: MIU,
                lambda: LAMBDA,
            },
            old_hessian_list: RefCell::<_>::new(vec![
                SMatrix::<f64, CO_NUM, CO_NUM>::zeros();
                p.n_prims
            ]),
            name: String::from(name),
            plane: p,

            circle: StaticCircle {
                keta: KETA,
                radius: 1.0,
                center: dvector![-1.5, 0.0],
            },
            circle2: StaticCircle {
                keta: KETA,
                radius: 1.0,
                center: dvector![1.5, -2.0],
            },
            x_tao: DVector::<f64>::zeros(1),
            g_vec,
        }
    }
}

pub fn main() {
    let problem = BouncingUpdateScenario::new("mybounce");
    let solver = NewtonSolver {
        max_iter: 30,
        epi: 1e-5,
    };
    let linearsolver = PivLU {};
    let linesearch = SimpleLineSearch {
        alpha: 0.9,
        tol: 1e-5,
        epi: 1.0,
    };
    let mut b = Scenario::new(problem, solver, linearsolver, linesearch);
    for _i in 0..100 {
        // b.mystep(false);
        b.step(true);
    }
}
