use autodiff::Hessian;
use mesh::{armadillo, Mesh3d, NeoHookean3d};
use nalgebra::{DMatrix, DVector, SMatrix, SVector};
use nalgebra_sparse::{
    ops::{serial::spadd_csr_prealloc, Op},
    CsrMatrix,
};
use optimization::{JacobianPre, LinearSolver, NewtonCG, NewtonSolver, Problem, SimpleLineSearch};
use std::cell::RefCell;
use thesis::scenarios::{Scenario, ScenarioProblem};

const FILENAME: &'static str = "armadillo_old.txt";
const COMMENT: &'static str = "naive projected newton";
const E: f64 = 1e5;
const NU: f64 = 0.33;
const MIU: f64 = E / (2.0 * (1.0 + NU));
const LAMBDA: f64 = (E * NU) / ((1.0 + NU) * (1.0 - 2.0 * NU));
const DT: f64 = 0.01;
const DIM: usize = 3;
const CO_NUM: usize = DIM * (DIM + 1);
const NFIXED_VERT: usize = 20;
const DAMP: f64 = 1.0;
const TOTAL_FRAME: usize = 200;

type EnergyType = NeoHookean3d;

pub struct BouncingUpdateScenario {
    armadillo: Mesh3d,
    dt: f64,
    name: String,
    energy: EnergyType,
    old_hessian_list: RefCell<Vec<SMatrix<f64, CO_NUM, CO_NUM>>>,
    old_elastic_hessian: RefCell<CsrMatrix<f64>>,
    x_tao: DVector<f64>,
    g_vec: DVector<f64>,
    mass: DMatrix<f64>,
    sparse_mass: CsrMatrix<f64>,
    index_list: Vec<SMatrix<usize, CO_NUM, CO_NUM>>,
}
impl BouncingUpdateScenario {
    fn elastic_my_hessian_new(
        &self,
        x: &DVector<f64>,
        old: &mut CsrMatrix<f64>,
        active_set: &std::collections::HashSet<usize>,
    ) {
        let update_triangle_list = active_set
            .iter()
            .map(|x| self.armadillo.vert_connected_prim_indices[x / DIM].clone())
            .flatten()
            .collect::<std::collections::HashSet<usize>>();

        for count in update_triangle_list {
            let indices = self.armadillo.get_indices(count);
            let mut vert_vec = SVector::<f64, CO_NUM>::zeros();
            vert_vec
                .iter_mut()
                .zip(indices.iter())
                .for_each(|(g_i, i)| *g_i = x[*i]);

            let energy: Hessian<CO_NUM> = self.armadillo.prim_energy(count, &self.energy, vert_vec);
            let new_small_hessian: SMatrix<f64, CO_NUM, CO_NUM> = energy.hessian();

            let diff_small_hessian = new_small_hessian - self.old_hessian_list.borrow()[count];

            self.old_hessian_list.borrow_mut()[count] = new_small_hessian;

            let old_values = old.values_mut();

            let diff_small_hessian_slice = diff_small_hessian.as_slice();
            let index_slice = self.index_list[count].as_slice();

            unsafe {
                for (v, i) in diff_small_hessian_slice
                    .chunks_exact(4)
                    .zip(index_slice.chunks_exact(4))
                {
                    *(old_values.get_unchecked_mut(i[0])) += v[0];
                    *(old_values.get_unchecked_mut(i[1])) += v[1];
                    *(old_values.get_unchecked_mut(i[2])) += v[2];
                    *(old_values.get_unchecked_mut(i[3])) += v[3];
                }
            }
        }
    }

    fn inertia_apply(&self, x: &DVector<f64>) -> f64 {
        let temp = x - &self.x_tao - &self.g_vec * (self.dt * self.dt);
        let res = temp.dot(&(&self.mass * &temp));
        res / 2.0
    }
    fn inertia_gradient(&self, x: &DVector<f64>) -> DVector<f64> {
        let res_grad = &self.mass * (x - &self.x_tao - &self.g_vec * (self.dt * self.dt));
        res_grad
    }
    fn inertia_hessian<'a>(&'a self, _x: &DVector<f64>) -> &'a DMatrix<f64> {
        // self.mass has already been divided by dt*dt when constructing it
        &self.mass
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
        // dense version of hessian matrix
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

        // sparse version of hessian matrix
        // let res = self.armadillo.elastic_hessian(x, &self.energy);
        // let mut res = CsrMatrix::from(&res);
        // for (r, c, v) in res.triplet_iter_mut() {
        //     if r < NFIXED_VERT * DIM || c < NFIXED_VERT * DIM {
        //         *v = 0.0;
        //     }
        // }
        // Some(res+ &self.sparse_mass)
    }
}

impl ScenarioProblem for BouncingUpdateScenario {
    fn initial_guess(&self) -> DVector<f64> {
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
        let energy = EnergyType {
            mu: MIU,
            lambda: LAMBDA,
        };
        let init_hessian = p.elastic_hessian(&p.verts, &energy);
        let init_hessian = CsrMatrix::from(&init_hessian);

        let mass = DMatrix::from_diagonal(&p.masss) / (DT * DT);
        let sparse_mass = CsrMatrix::from(&mass);

        let mut old_hessian_list = Vec::<SMatrix<f64, CO_NUM, CO_NUM>>::new();
        let mut index_list = Vec::<SMatrix<usize, CO_NUM, CO_NUM>>::new();
        for i in 0..p.n_prims {
            let mut vert_vec = SVector::<f64, CO_NUM>::zeros();
            let indices = p.get_indices(i);

            vert_vec
                .iter_mut()
                .zip(indices.iter())
                .for_each(|(g_i, i)| *g_i = p.verts[*i]);
            let energy: Hessian<CO_NUM> = p.prim_energy(i, &energy, vert_vec);
            old_hessian_list.push(energy.hessian());
            let mut index_matrix = SMatrix::<[usize; 2], CO_NUM, CO_NUM>::default();
            for i in 0..CO_NUM {
                for j in 0..CO_NUM {
                    index_matrix[(i, j)] = [indices[i], indices[j]];
                }
            }
        }

        Self {
            dt: DT,
            energy,
            name: String::from(name),
            old_hessian_list: RefCell::<_>::new(old_hessian_list),
            old_elastic_hessian: RefCell::<_>::new(init_hessian),
            mass,
            sparse_mass,
            index_list,
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
        epi: 1e-3,
    };
    let linearsolver = NewtonCG::<JacobianPre<CsrMatrix<f64>>>::new();
    let linesearch = SimpleLineSearch {
        alpha: 0.9,
        tol: 1e-5,
        epi: 0.0,
    };
    let mut b = Scenario::new(problem, solver, linearsolver, linesearch, FILENAME, COMMENT);

    for _i in 0..TOTAL_FRAME {
        println!("{}", _i);
        b.step();
    }
}
