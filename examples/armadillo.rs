use autodiff::Hessian;
use matrixcompare::assert_matrix_eq;
use mesh::{armadillo, Mesh3d, NeoHookean2d};
use nalgebra::{ComplexField, DMatrix, DVector, SMatrix, SVector};
use nalgebra_sparse::{
    ops::{
        serial::{spadd_csr_prealloc, spmm_csc_prealloc},
        Op,
    },
    pattern::SparsityPattern,
    CsrMatrix,
};
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
    old_elastic_hessian: RefCell<CsrMatrix<f64>>,
    hessian_pattern: SparsityPattern,
    x_tao: DVector<f64>,
    g_vec: DVector<f64>,
    mass: DMatrix<f64>,
    sparse_mass: CsrMatrix<f64>,
}
impl BouncingUpdateScenario {
    fn elastic_my_hessian_old(
        &self,
        x: &DVector<f64>,
        active_set: &std::collections::HashSet<usize>,
    ) -> DMatrix<f64> {
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
    fn elastic_my_hessian_new(
        &self,
        x: &DVector<f64>,
        old: &mut CsrMatrix<f64>,
        active_set: &std::collections::HashSet<usize>,
    ) {
        // let another_active_set = active_set
        //     .iter()
        //     .map(|x| x / DIM)
        //     .collect::<HashSet<usize>>();
        let update_triangle_list = active_set
            .iter()
            .map(|x| self.armadillo.vert_connected_prim_indices[*x].clone())
            .flatten()
            .collect::<HashSet<usize>>();

        for i in 0..self.armadillo.n_prims {
            let energy: Hessian<CO_NUM>;
            let hessian: SMatrix<f64,CO_NUM,CO_NUM>;
            let indices = self.armadillo.get_indices(i);
            if update_triangle_list.contains(&i) {
                let mut vert_vec = SVector::<f64, CO_NUM>::zeros();
                
                vert_vec
                    .iter_mut()
                    .zip(indices.iter())
                    .for_each(|(g_i, i)| *g_i = x[*i]);
                energy = self.armadillo.prim_energy(i, &self.energy, vert_vec);
                hessian = energy.hessian();
                self.old_hessian_list.borrow_mut()[i] = hessian;
            }else{
                hessian = self.old_hessian_list.borrow()[i];

            }
                for i in 0..CO_NUM {
                    for j in 0..CO_NUM {
                        if indices[i] < NFIXED_VERT * DIM || indices[j] < NFIXED_VERT * DIM {
                            continue;
                        }
                        // if !active_set.contains(&(indices[i]/DIM)) && !active_set.contains(&(indices[j]/DIM)) {
                        //     continue;
                        // }
                        let entry = old.index_entry_mut(indices[i], indices[j]);
                        match entry {
                            nalgebra_sparse::SparseEntryMut::NonZero(refe) => {
                                *refe += hessian[(i, j)];
                            }
                            nalgebra_sparse::SparseEntryMut::Zero => {
                                panic!("this shouldn't happent");
                            }
                        }
                    }
                }
            // }
            // else {
            //     let mut vert_vec = SVector::<f64, CO_NUM>::zeros();
            //     let indices = self.armadillo.get_indices(i);
            //     vert_vec
            //         .iter_mut()
            //         .zip(indices.iter())
            //         .for_each(|(g_i, i)| *g_i = x[*i]);
            //     let energy: Hessian<CO_NUM> = self.armadillo.prim_energy(i, &self.energy, vert_vec);
            //     let hessian = energy.hessian();
            //     for i in 0..CO_NUM {
            //         for j in 0..CO_NUM {
            //             if indices[i] < NFIXED_VERT * DIM || indices[j] < NFIXED_VERT * DIM {
            //                 continue;
            //             }
            //             // if !active_set.contains(&indices[i]) && !active_set.contains(&indices[j]) {
            //             //     continue;
            //             // }
            //             let entry = old.index_entry_mut(indices[i], indices[j]);
            //             match entry {
            //                 nalgebra_sparse::SparseEntryMut::NonZero(refe) => {
            //                     *refe += hessian[(i, j)];
            //                 }
            //                 nalgebra_sparse::SparseEntryMut::Zero => {
            //                     panic!("this shouldn't happent");
            //                 }
            //             }
            //         }
            //     }
            // }
        }
    }

    fn inertia_apply(&self, x: &DVector<f64>) -> f64 {
        let temp = x - &self.x_tao - &self.g_vec * (self.dt * self.dt);
        let res = temp.dot(&(&self.mass * &temp));
        // res / (2.0 * self.dt * self.dt)
        res / 2.0
    }
    fn inertia_gradient(&self, x: &DVector<f64>) -> DVector<f64> {
        let res_grad = &self.mass * (x - &self.x_tao - &self.g_vec * (self.dt * self.dt));
        // res_grad / (self.dt * self.dt)
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
    fn my_gradient(
        &self,
        x: &DVector<f64>,
    ) -> (
        Option<DVector<f64>>,
        Option<std::collections::HashSet<usize>>,
    ) {
        let res = self.gradient(x).unwrap();
        let mut active_set = std::collections::HashSet::<usize>::new();
        let max = 0.1 * res.amax();
        for (i, r) in res.iter().enumerate() {
            if r.abs() > max {
                active_set.insert(i / DIM);
            }
        }
        (Some(res), Some(active_set))
    }

    fn my_hessian(
        &self,
        x: &DVector<f64>,
        active_set: &std::collections::HashSet<usize>,
    ) -> Option<<Self as Problem>::HessianType> {
        let mut old_matrix = self.old_elastic_hessian.borrow_mut();
        // let active_set: std::collections::HashSet<usize> = (0..x.len()).collect();
        // clear active entries
        for (r, c, v) in old_matrix.triplet_iter_mut() {
            // if active_set.contains(&(c / DIM)) || active_set.contains(&(r / DIM)) {
                *v = 0.0;
                // continue;
            // } else if r < NFIXED_VERT * DIM || c < NFIXED_VERT * DIM {
                // *v = 0.0;
                // continue;
            // }
        }

        self.elastic_my_hessian_new(x, &mut old_matrix, &active_set);

        Some(&self.sparse_mass + old_matrix.clone())

        // let mut res = DMatrix::<f64>::zeros(x.len(), x.len());
        // res = res + self.inertia_hessian(x);
        // res = res + self.elastic_my_hessian_old(x, active_set);

        // let mut slice = res.index_mut((0..NFIXED_VERT * DIM, NFIXED_VERT * DIM..));
        // for i in slice.iter_mut() {
        //     *i = 0.0;
        // }
        // let mut slice = res.index_mut((NFIXED_VERT * DIM.., 0..NFIXED_VERT * DIM));
        // for i in slice.iter_mut() {
        //     *i = 0.0;
        // }
        // let res = CsrMatrix::from(&res);
        // Some(res)
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
        let energy = EnergyType {
            mu: MIU,
            lambda: LAMBDA,
        };
        let init_hessian = p.elastic_hessian(&p.verts, &energy);
        let init_hessian = CsrMatrix::from(&init_hessian);
        let hessian_pattern = init_hessian.pattern().clone();
        let mass = DMatrix::from_diagonal(&p.masss) / (DT * DT);
        let sparse_mass = CsrMatrix::from(&mass);
        let mut old_hessian_list = Vec::<SMatrix<f64, CO_NUM, CO_NUM>>::new();
        for i in 0..p.n_prims {
            let mut vert_vec = SVector::<f64, CO_NUM>::zeros();
            let indices = p.get_indices(i);

            vert_vec
                .iter_mut()
                .zip(indices.iter())
                .for_each(|(g_i, i)| *g_i = p.verts[*i]);
            let energy: Hessian<CO_NUM> = p.prim_energy(i, &energy, vert_vec);
            old_hessian_list.push(energy.hessian());
        }
        Self {
            dt: DT,
            energy,
            name: String::from(name),
            old_hessian_list: RefCell::<_>::new(old_hessian_list),
            old_elastic_hessian: RefCell::<_>::new(init_hessian),
            hessian_pattern,
            mass,
            sparse_mass,

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
        b.mystep(true);
        // b.step(true);
    }
}

// possible bug: H_{ii} part has problems when constrcut my_elastic
