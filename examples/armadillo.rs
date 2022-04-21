use mesh::{armadillo, Mesh3d};
use nalgebra::{DMatrix, DVector};
use nalgebra_sparse::CsrMatrix;
use optimization::{JacobianPre, LinearSolver, NewtonCG, NewtonSolver, Problem, SimpleLineSearch};
use thesis::scenarios::{Scenario, ScenarioProblem};
mod armadillopara;
use armadillopara::*;
const FILENAME: &'static str = "armadillo_projected.txt";
const COMMENT: &'static str = "naive projected newton";

pub struct BouncingUpdateScenario {
    armadillo: Mesh3d,
    dt: f64,
    name: String,
    energy: EnergyType,
    x_tao: DVector<f64>,
    g_vec: DVector<f64>,
    mass: CsrMatrix<f64>,
}
impl BouncingUpdateScenario {
    fn inertia_apply(&self, x: &DVector<f64>) -> f64 {
        let temp = x - &self.x_tao - &self.g_vec * (self.dt * self.dt);
        let res = temp.dot(&(&self.mass * &temp));
        res / 2.0
    }
    fn inertia_gradient(&self, x: &DVector<f64>) -> DVector<f64> {
        let res_grad = &self.mass * (x - &self.x_tao - &self.g_vec * (self.dt * self.dt));
        res_grad
    }
    fn inertia_hessian<'a>(&'a self, _x: &DVector<f64>) -> &'a CsrMatrix<f64> {
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
        let mut elastic_hessian =
            CsrMatrix::from(&self.armadillo.elastic_hessian_projected(x, &self.energy));

        for (i, j, k) in elastic_hessian.triplet_iter_mut() {
            if i < NFIXED_VERT * DIM || j < NFIXED_VERT * DIM {
                *k = 0.0;
            }
        }

        Some(self.inertia_hessian(x) + elastic_hessian)
    }

    fn hessian_inverse_mut<'a>(
        &'a mut self,
        _x: &DVector<f64>,
        // ) -> nalgebra_sparse::factorization::CscCholesky<f64>;
    ) -> &'a nalgebra_sparse::CscMatrix<f64> {
        todo!()
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

        let mass = DMatrix::from_diagonal(&p.masss) / (DT * DT);
        let mass = CsrMatrix::from(&mass);
        Self {
            dt: DT,
            energy,
            name: String::from(name),
            mass,
            armadillo: p,

            x_tao: DVector::<f64>::zeros(1),
            g_vec,
        }
    }
}

fn main() {
    let problem = BouncingUpdateScenario::new("armadillotru");

    let solver = NewtonSolver {
        max_iter: 300,
        epi: 1e-3,
    };
    let linearsolver = NewtonCG::<JacobianPre<CsrMatrix<f64>>>::new();
    let linesearch = SimpleLineSearch {
        alpha: 0.9,
        tol: 1e-5,
        epi: 0.0,
    };
    let mut b = Scenario::new(problem, solver, linearsolver, linesearch, FILENAME, COMMENT);

    let start = std::time::Instant::now();
    for _i in 0..TOTAL_FRAME {
        println!("{}", _i);
        b.step();
    }
    println!("{}", start.elapsed().as_secs_f32());
}
