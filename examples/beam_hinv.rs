use autodiff::Hessian;
use mesh::*;
use nalgebra::{DMatrix, DVector, SMatrix, SVector};
use nalgebra_sparse::{factorization::CscCholesky, CscMatrix, CsrMatrix};
use optimization::*;
use thesis::scenarios::{Scenario, ScenarioProblem};

const FILENAME: &'static str = "beamnew.txt";
const COMMENT: &'static str = "modifited";
const E: f64 = 1e7;
const NU: f64 = 0.4;
const MIU: f64 = E / (2.0 * (1.0 + NU));
const LAMBDA: f64 = (E * NU) / ((1.0 + NU) * (1.0 - 2.0 * NU));
const DT: f64 = 1.0 / 60.0;
const DIM: usize = 2;
const CO_NUM: usize = DIM * (DIM + 1);

const NFIXED_VERT: usize = 20;
#[allow(non_upper_case_globals)]
const c: usize = 80;
const DAMP: f64 = 1.0;
const TOTAL_FRAME: usize = 500;

const ACTIVE_SET_EPI: f64 = 0.01;

type EnergyType = NeoHookean2d;

pub struct BeamScenario {
    beam: Mesh2d,
    dt: f64,
    name: String,
    energy: EnergyType,
    x_tao: DVector<f64>,
    g_vec: DVector<f64>,
    mass: DMatrix<f64>,
    active_set: std::collections::HashSet<usize>,
    hessian_list: Vec<SMatrix<f64, CO_NUM, CO_NUM>>,
    hessian_inverse: DMatrix<f64>,
}
impl BeamScenario {
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
impl Problem for BeamScenario {
    type HessianType = CsrMatrix<f64>;
    fn apply(&self, x: &DVector<f64>) -> f64 {
        let mut res = 0.0;
        res += self.inertia_apply(x);
        res += self.beam.elastic_apply(x, &self.energy);
        res
    }

    fn gradient(&self, x: &DVector<f64>) -> Option<DVector<f64>> {
        let mut res = DVector::<f64>::zeros(x.len());
        res += self.inertia_gradient(x);
        res += self.beam.elastic_gradient(x, &self.energy);

        let mut slice = res.index_mut((0..NFIXED_VERT * DIM, 0));
        for i in slice.iter_mut() {
            *i = 0.0;
        }
        Some(res)
    }

    fn gradient_mut(&mut self, x: &DVector<f64>) -> Option<DVector<f64>> {
        let res = self.gradient(x).unwrap();
        // reset active set
        {
            self.active_set.clear();
            let boundary = ACTIVE_SET_EPI * res.amax();
            res.iter().enumerate().for_each(|(i, x)| {
                if x.abs() > boundary {
                    self.active_set.insert(i / DIM);
                }
            });
        }

        Some(res)
    }

    fn hessian(&self, x: &DVector<f64>) -> Option<Self::HessianType> {
        // dense version of hessian matrix
        let mut res = DMatrix::<f64>::zeros(x.len(), x.len());
        res = res + self.inertia_hessian(x);
        res += self.beam.elastic_hessian(x, &self.energy);

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
    fn hessian_inverse_mut(&mut self, x: &DVector<f64>) -> Option<DMatrix<f64>> {
        let mut res = DMatrix::<f64>::zeros(x.len(), x.len());
        res = res + self.inertia_hessian(x);

        let update_triangle_list = self
            .active_set
            .iter()
            .map(|x| self.beam.vert_connected_prim_indices[*x].clone())
            // .flatten()
            // .map(|x| self.beam.prim_connected_vert_indices[x].clone())
            // .flatten()
            // .map(|x| self.beam.vert_connected_prim_indices[x].clone())
            .flatten()
            .collect::<std::collections::HashSet<usize>>();
        for i in update_triangle_list {
            println!("?");
            let mut vert_vec = SVector::<f64, CO_NUM>::zeros();
            let indices = self.beam.get_indices(i);

            vert_vec
                .iter_mut()
                .zip(indices.iter())
                .for_each(|(g_i, i)| *g_i = x[*i]);
            let energy: Hessian<CO_NUM> = self.beam.prim_energy(i, &self.energy, vert_vec);
            let energy_hessian = energy.hessian();
            let old_energy_hessian = self.hessian_list[i];

            // rank 4 matrix
            let diff = energy_hessian - old_energy_hessian;

            let decom = diff.symmetric_eigen();
            let eigenvalues = decom.eigenvalues;

            let eigenvectors = decom.eigenvectors;
            // TODO: hard-coded 4x4 matrix
            let C_inv = SMatrix::<f64, 4, 4>::from_diagonal(&SVector::<f64, 4>::new(
                1.0 / eigenvalues[0],
                1.0 / eigenvalues[1],
                1.0 / eigenvalues[2],
                1.0 / eigenvalues[3],
            ));

            let U = eigenvectors.fixed_columns::<4>(0);
            //  diff = &U * &C * U.transpose()

            let middle = C_inv + U.transpose() * &self.hessian_inverse * U;

            let middle_inv = middle.try_inverse().unwrap();

            self.hessian_inverse -=
                &self.hessian_inverse * &U * middle_inv * U.transpose() * &self.hessian_inverse;

            // update the hessian list
            self.hessian_list[i] = energy_hessian;
        }

        // res += &self.init_hessian;

        // let mut slice = res.index_mut((0..NFIXED_VERT * DIM, NFIXED_VERT * DIM..));
        // for i in slice.iter_mut() {
        //     *i = 0.0;
        // }
        // let mut slice = res.index_mut((NFIXED_VERT * DIM.., 0..NFIXED_VERT * DIM));
        // for i in slice.iter_mut() {
        //     *i = 0.0;
        // }
        // Some(CsrMatrix::from(&res))
        let hessian = self.hessian(x).unwrap();
        let hessian = CscMatrix::from(&hessian);
        let solver = nalgebra_sparse::factorization::CscCholesky::factor(&hessian);
        let hessian_inverse = solver
            .unwrap()
            .solve(&DMatrix::<f64>::identity(x.len(), x.len()));
        Some(hessian_inverse)
    }
}

impl ScenarioProblem for BeamScenario {
    fn initial_guess(&self) -> DVector<f64> {
        self.beam.verts.clone()
    }
    fn set_all_vertices_vector(&mut self, vertices: DVector<f64>) {
        let velocity = DAMP * ((&vertices - &self.beam.verts) / self.dt);
        self.beam.velos = velocity;
        self.beam.verts = vertices;
    }
    fn save_to_file(&self, frame: usize) {
        self.beam
            .save_to_obj(format!("output/{}{}.obj", self.name, frame));
    }
    fn frame_init(&mut self) {
        self.x_tao = &self.beam.verts + self.dt * &self.beam.velos;
    }
    fn frame_end(&mut self) {
        // self.frame+=1;
    }
}

impl BeamScenario {
    pub fn new(name: &str) -> Self {
        let mut p = plane(NFIXED_VERT, c, Some(0.125), Some(0.125), None);

        // init velocity
        for i in 0..c {
            for j in 0..NFIXED_VERT {
                p.velos[DIM * (i * NFIXED_VERT + j)] =
                    -1.0 * (i as f64) * (i as f64) * (i as f64 / 20.0) * 0.125 * 0.125 * 0.125;
            }
        }

        let mut g_vec = DVector::zeros(DIM * p.n_verts);
        for i in NFIXED_VERT..p.n_verts {
            g_vec[DIM * i] = -9.8;
        }
        let energy = EnergyType {
            mu: MIU,
            lambda: LAMBDA,
        };

        let mass = DMatrix::from_diagonal(&p.masss) / (DT * DT);

        let init_hessian = p.elastic_hessian(&p.verts, &energy);

        let hessian_inverse = init_hessian + &mass;
        let hessian_inverse = CscMatrix::<f64>::from(&hessian_inverse);
        let solver = nalgebra_sparse::factorization::CscCholesky::factor(&hessian_inverse);
        let hessian_inverse = solver
            .unwrap()
            .solve(&DMatrix::<f64>::identity(p.n_verts * DIM, p.n_verts * DIM));

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

            // old_hessian_list.push(SMatrix::<f64,6,6>::zeros());
        }

        let scenario = Self {
            dt: DT,
            energy,
            name: String::from(name),
            mass,
            beam: p,
            x_tao: DVector::<f64>::zeros(1),
            g_vec,
            active_set: std::collections::HashSet::<usize>::new(),
            hessian_list: old_hessian_list,
            hessian_inverse,
        };
        scenario
    }
}

fn main() {
    let problem = BeamScenario::new("beam");

    let solver = NewtonInverseSolver {
        max_iter: 30,
        epi: 1e-5,
    };
    let mut linearsolver = NewtonCG::<JacobianPre<CsrMatrix<f64>>>::new();
    linearsolver.tol = 1e-10;
    let linesearch = SimpleLineSearch {
        alpha: 0.9,
        tol: 0.01,
        epi: 1e-7,
    };
    let mut b = Scenario::new(problem, solver, linearsolver, linesearch, FILENAME, COMMENT);

    for _i in 0..TOTAL_FRAME {
        println!("running frame: {}", _i);
        b.step();
    }
}
