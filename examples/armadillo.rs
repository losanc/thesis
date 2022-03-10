use autodiff::{Hessian};
use mesh::{armadillo, Mesh3d, NeoHookean2d};
use nalgebra::{DMatrix, DVector, SMatrix, SVector};
use nalgebra_sparse::CsrMatrix;
use optimization::{JacobianPre, LinearSolver, NewtonCG, NewtonSolver, Problem, SimpleLineSearch};
use std::{cell::RefCell, collections::HashSet};
use thesis::{
    my_newton::MyProblem,
    scenarios::{Scenario, ScenarioProblem},
};

const E: f64 = 1e6;
const NU: f64 = 0.33;
const MIU: f64 = E / (2.0 * (1.0 + NU));
const LAMBDA: f64 = (E * NU) / ((1.0 + NU) * (1.0 - 2.0 * NU));
const DT: f64 = 0.01;
const DIM: usize = 3;
const CO_NUM: usize = DIM * (DIM + 1);
const NFIXED_VERT: usize = 20;
const DAMP: f64 = 1.0;

type EnergyType = NeoHookean2d;

pub struct BouncingUpdateScenario {
    armadillo: Mesh3d,
    dt: f64,
    name: String,
    energy: EnergyType,
    old_hessian_list: RefCell<Vec<SMatrix<f64, CO_NUM, CO_NUM>>>,
    x_tao: DVector<f64>,
    g_vec: DVector<f64>,
    mass: DMatrix<f64>,
}
impl BouncingUpdateScenario {

    fn elastic_my_hessian(&self, x: &DVector<f64>, active_set: &[usize]) -> DMatrix<f64> {
        let mut res = DMatrix::zeros(x.len(), x.len());
        let active_set = active_set
            .iter()
            .map(|x| x / DIM)
            .collect::<HashSet<usize>>();
        let update_triangle_list = active_set
            .iter()
            .map(|x| self.armadillo.vert_connected_prim_indices[*x].clone())
            .flatten()
            .collect::<HashSet<usize>>();

        for i in 0..self.armadillo.n_prims {
            let small_hessian;
            let indices = self.armadillo.get_indices(i);
            if update_triangle_list.contains(&i) {
                let mut vert_vec = SVector::<f64, CO_NUM>::zeros();
                vert_vec
                    .iter_mut()
                    .zip(indices.iter())
                    .for_each(|(g_i, i)| *g_i = x[*i]);
                let energy: Hessian<CO_NUM> = self.armadillo.prim_energy(i, &self.energy, vert_vec);
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
}
impl Problem for BouncingUpdateScenario {
    type HessianType = CsrMatrix<f64>;
    fn apply(&self, x: &DVector<f64>) -> f64 {
        let mut res = 0.0;
        res += self.inertia_apply(x);
        res += self.armadillo.elastic_apply(x, &self.energy);
        res
    }

    fn gradient(&self, x: &DVector<f64>) -> Option<DVector<f64>> {
        let mut res = DVector::<f64>::zeros(x.len());
        res += self.inertia_gradient(x);
        res += self.armadillo.elastic_gradient(x, &self.energy);

        let mut slice = res.index_mut((0..NFIXED_VERT * DIM, 0));
        for i in slice.iter_mut() {
            *i = 0.0;
        }
        Some(res)
    }

    fn hessian(&self, x: &DVector<f64>) -> Option<Self::HessianType> {
        let mut res = DMatrix::<f64>::zeros(x.len(), x.len());
        res = res + self.inertia_hessian(x);
        res += self.armadillo.elastic_hessian(x, &self.energy);

        let mut slice = res.index_mut((0..NFIXED_VERT * DIM, NFIXED_VERT * DIM..));
        for i in slice.iter_mut() {
            *i = 0.0;
        }
        let mut slice = res.index_mut((NFIXED_VERT * DIM.., 0..NFIXED_VERT * DIM));
        for i in slice.iter_mut() {
            *i = 0.0;
        }
        Some(CsrMatrix::from(&res))
    }
}

impl MyProblem for BouncingUpdateScenario {
    fn my_gradient(&self, x: &DVector<f64>) -> (Option<DVector<f64>>, Option<Vec<usize>>) {
        let res = self.gradient(x).unwrap();
        let mut active_set = Vec::<usize>::new();
        let max = 0.1 * res.amax();
        for (i, r) in res.iter().enumerate() {
            if r.abs() > max {
                active_set.push(i);
            }
        }
        (Some(res), Some(active_set))
    }

    fn my_hessian(
        &self,
        x: &DVector<f64>,
        active_set: &[usize],
    ) -> Option<<Self as Problem>::HessianType> {
        let mut res = DMatrix::<f64>::zeros(x.len(), x.len());
        res = res + self.inertia_hessian(x);
        res = res + self.elastic_my_hessian(x, active_set);

        let mut slice = res.index_mut((0..NFIXED_VERT * DIM, NFIXED_VERT * DIM..));
        for i in slice.iter_mut() {
            *i = 0.0;
        }
        let mut slice = res.index_mut((NFIXED_VERT * DIM.., 0..NFIXED_VERT * DIM));
        for i in slice.iter_mut() {
            *i = 0.0;
        }
        let res = CsrMatrix::from(&res);
        Some(res)
    }
}

impl ScenarioProblem for BouncingUpdateScenario {
    fn initial_guess(&self) -> DVector<f64> {
        // print!("{:?}",CsrMatrix::from(&self.armadillo.elastic_hessian(&self.armadillo.verts, &self.energy)));
        self.armadillo.verts.clone()
      
    }
    fn set_all_vertices_vector(&mut self, vertices: DVector<f64>) {
        let velocity = DAMP * ((&vertices - &self.armadillo.verts) / self.dt);
        self.armadillo.velos = velocity;
        self.armadillo.verts = vertices;
    }
    fn save_to_file(&self, frame: usize) {
        self.armadillo
            .save_to_obj(format!("output/{}{}.obj", self.name, frame));
    }
    fn frame_init(&mut self) {
        self.x_tao = &self.armadillo.verts + self.dt * &self.armadillo.velos;
    }
    fn frame_end(&mut self) {
        // self.frame+=1;
    }
}

impl BouncingUpdateScenario {
    pub fn new(name: &str) -> Self {
        let p = armadillo();
        let mut g_vec = DVector::zeros(DIM * p.n_verts);
        for i in NFIXED_VERT..p.n_verts {
            g_vec[DIM * i + 1] = -9.8;
        }

        Self {
            dt: DT,
            energy: EnergyType {
                mu: MIU,
                lambda: LAMBDA,
            },
            name: String::from(name),
            old_hessian_list: RefCell::<_>::new(vec![
                SMatrix::<f64, CO_NUM, CO_NUM>::zeros();
                p.n_prims
            ]),
            mass: DMatrix::from_diagonal(&p.masss),
            armadillo: p,

            x_tao: DVector::<f64>::zeros(1),
            g_vec,
        }
    }
}

fn main() {
    let problem = BouncingUpdateScenario::new("armadillotru");
    let solver = NewtonSolver {
        max_iter: 30,
        epi: 0.1,
    };
    let linearsolver = NewtonCG::<JacobianPre<CsrMatrix<f64>>>::new();
    // let linearsolver = NewtonCG::<NoPre<DMatrix<f64>>>::new();
    let linesearch = SimpleLineSearch {
        alpha: 0.9,
        tol: 1e-5,
        epi: 1.0,
    };
    // let linesearch = NoLineSearch {};
    let mut b = Scenario::new(problem, solver, linearsolver, linesearch);
    for _i in 0..100 {
        println!("{}", _i);
        // b.mystep(true);
        b.step(true);
    }
}
