use autodiff::Hessian;
use mesh::*;
use nalgebra::{DMatrix, DVector, SMatrix, SVector};
use nalgebra_sparse::CscMatrix;
use optimization::*;
use thesis::scenarios::{Scenario, ScenarioProblem};
mod circlepara;
use circlepara::*;

pub const FILENAME: &'static str = "circlenew.txt";
pub const COMMENT: &'static str = "modiifed";

pub struct CircleScenario {
    beam: Mesh2d,
    dt: f64,
    name: String,
    energy: EnergyType,
    x_tao: DVector<f64>,
    g_vec: DVector<f64>,
    mass: DMatrix<f64>,
    active_set: std::collections::HashSet<usize>,
    hessian_list: Vec<SMatrix<f64, CO_NUM, CO_NUM>>,
    init_hessian: DMatrix<f64>,
}
impl CircleScenario {
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

impl Problem for CircleScenario {
    type HessianType = CscMatrix<f64>;
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

    fn hessian_mut(&mut self, x: &DVector<f64>) -> Option<Self::HessianType> {
        // dense version of hessian matrix
        let mut res = DMatrix::<f64>::zeros(x.len(), x.len());
        res = res + self.inertia_hessian(x);

        let update_triangle_list = self
            .active_set
            .iter()
            .map(|x| self.beam.vert_connected_prim_indices[*x].clone())
            .flatten()
            .collect::<std::collections::HashSet<usize>>();

        for i in update_triangle_list {
            let mut vert_vec = SVector::<f64, CO_NUM>::zeros();
            let indices = self.beam.get_indices(i);

            vert_vec
                .iter_mut()
                .zip(indices.iter())
                .for_each(|(g_i, i)| *g_i = x[*i]);
            let energy: Hessian<CO_NUM> = self.beam.prim_energy(i, &self.energy, vert_vec);
            let energy_hessian = energy.hessian();

            let mut eigendecomposition = energy_hessian.symmetric_eigen();
            for eigenvalue in eigendecomposition.eigenvalues.iter_mut() {
                if *eigenvalue < 0.0 {
                    *eigenvalue = 0.0;
                }
            }
            let energy_hessian = eigendecomposition.recompose();

            let old_energy_hessian = self.hessian_list[i];
            let diff = energy_hessian - old_energy_hessian;
            // update global hessian
            for i in 0..CO_NUM {
                for j in 0..CO_NUM {
                    self.init_hessian[(indices[i], indices[j])] += diff[(i, j)];
                }
            }
            // update the hessian list
            self.hessian_list[i] = energy_hessian;
        }

        res += &self.init_hessian;

        Some(CscMatrix::from(&res))
    }

    fn hessian_inverse_mut<'a>(&'a mut self, _x: &DVector<f64>) -> &'a CscMatrix<f64> {
        todo!()
    }
}

impl ScenarioProblem for CircleScenario {
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

impl CircleScenario {
    pub fn new(name: &str) -> Self {
        let mut p = circle(R, RES, None);

        for i in 0..p.n_verts {
            p.verts[DIM * i] *= 2.0;
            p.verts[DIM * i + 1] *= 0.5;
        }
        for i in 0..p.n_verts {
            p.velos[DIM * i] = 0.1 * (i as f64) * p.verts[DIM * i];
            p.velos[DIM * i + 1] = 0.1 * (i as f64) * p.verts[DIM * i + 1];
        }

        let g_vec = DVector::zeros(DIM * p.n_verts);
        // for i in NFIXED_VERT..p.n_verts {
        //     g_vec[DIM * i] = -9.8;
        // }
        let energy = EnergyType {
            mu: MIU,
            lambda: LAMBDA,
        };

        let mass = DMatrix::from_diagonal(&p.masss) / (DT * DT);

        let init_hessian = p.elastic_hessian_projected(&p.verts, &energy);

        let mut old_hessian_list = Vec::<SMatrix<f64, CO_NUM, CO_NUM>>::new();
        for i in 0..p.n_prims {
            let mut vert_vec = SVector::<f64, CO_NUM>::zeros();
            let indices = p.get_indices(i);

            vert_vec
                .iter_mut()
                .zip(indices.iter())
                .for_each(|(g_i, i)| *g_i = p.verts[*i]);
            let energy: Hessian<CO_NUM> = p.prim_energy(i, &energy, vert_vec);

            let energy_hessian = energy.hessian();
            let mut eigendecomposition = energy_hessian.symmetric_eigen();
            for eigenvalue in eigendecomposition.eigenvalues.iter_mut() {
                if *eigenvalue < 0.0 {
                    *eigenvalue = 0.0;
                }
            }
            let energy_hessian = eigendecomposition.recompose();
            old_hessian_list.push(energy_hessian);
        }

        Self {
            dt: DT,
            energy,
            name: String::from(name),
            mass,
            beam: p,
            x_tao: DVector::<f64>::zeros(1),
            g_vec,
            active_set: std::collections::HashSet::<usize>::new(),
            hessian_list: old_hessian_list,
            init_hessian,
        }
    }
}

fn main() {
    let problem = CircleScenario::new("circlenew");

    let solver = NewtonSolverMut {
        max_iter: 500,
        epi: 1e-5,
    };
    // let mut linearsolver = NewtonCG::<JacobianPre<CsrMatrix<f64>>>::new();
    let linearsolver = CscCholeskySolver {};
    let linesearch = SimpleLineSearch {
        alpha: 0.9,
        tol: 0.01,
        epi: 1e-7,
    };
    let mut b = Scenario::new(problem, solver, linearsolver, linesearch, FILENAME, COMMENT);
    let start = std::time::Instant::now();
    for _i in 0..TOTAL_FRAME {
        println!("running frame: {}", _i);
        b.step();
    }
    let duration = start.elapsed().as_secs_f32();
    println!("time spent {duration}");
}
