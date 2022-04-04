use mesh::{plane, Mesh2d, NeoHookean2d};
use nalgebra::{DMatrix, DVector};
use nalgebra_sparse::CsrMatrix;
use optimization::{linesearch, NewtonSolver, NoLineSearch, PivLUCsr, Problem, SimpleLineSearch};
use thesis::{
    my_newton::MyProblem,
    scenarios::{Scenario, ScenarioProblem},
};

const E: f64 = 1e6;
const NU: f64 = 0.33;
const MIU: f64 = E / (2.0 * (1.0 + NU));
const LAMBDA: f64 = (E * NU) / ((1.0 + NU) * (1.0 - 2.0 * NU));
const DT: f64 = 0.01;
const DIM: usize = 2;
const NFIXED_VERT: usize = 10;
const DAMP: f64 = 1.0;
const TOTAL_FRAME: usize = 200;

type EnergyType = NeoHookean2d;

pub struct BeamScenario {
    beam: Mesh2d,
    dt: f64,
    name: String,
    energy: EnergyType,
    x_tao: DVector<f64>,
    g_vec: DVector<f64>,
    mass: DMatrix<f64>,
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
}

impl MyProblem for BeamScenario {}

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
        let p = plane(NFIXED_VERT, 40, None);
        let mut g_vec = DVector::zeros(DIM * p.n_verts);
        for i in NFIXED_VERT..p.n_verts {
            g_vec[DIM * i] = -9.8;
        }
        let energy = EnergyType {
            mu: MIU,
            lambda: LAMBDA,
        };

        let mass = DMatrix::from_diagonal(&p.masss) / (DT * DT);

        Self {
            dt: DT,
            energy,
            name: String::from(name),
            mass,
            beam: p,
            x_tao: DVector::<f64>::zeros(1),
            g_vec,
        }
    }
}

fn main() {
    let problem = BeamScenario::new("beam");

    let solver = NewtonSolver {
        max_iter: 30,
        epi: 1e-3,
    };
    let linearsolver = PivLUCsr {};
    let linesearch = NoLineSearch {};
    let mut b = Scenario::new(problem, solver, linearsolver, linesearch);

    for _i in 0..TOTAL_FRAME {
        println!("{}", _i);
        b.step(true);
    }
}
