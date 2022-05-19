use mesh::*;
use nalgebra::{DMatrix, DVector};
use nalgebra_sparse::{CscMatrix, CsrMatrix};
use optimization::*;
use thesis::scenarios::{Scenario, ScenarioProblem};
pub mod beampara;
use beampara::*;

pub const FILENAME: &'static str = "beam";
pub const COMMENT: &'static str = "vanilla projected newton";

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

        let mut slice = res.index_mut((0..ROW * DIM, 0));
        for i in slice.iter_mut() {
            *i = 0.0;
        }
        Some(res)
    }
    fn gradient_mut(&mut self, x: &DVector<f64>) -> Option<DVector<f64>> {
        self.gradient(x)
    }

    fn hessian_mut(&mut self, x: &DVector<f64>) -> (Option<Self::HessianType>, usize) {
        // dense version of hessian matrix
        let mut res = DMatrix::<f64>::zeros(x.len(), x.len());
        res = res + self.inertia_hessian(x);
        res += self.beam.elastic_hessian_projected(x, &self.energy);

        let mut slice = res.index_mut((0..ROW * DIM, ROW * DIM..));
        for i in slice.iter_mut() {
            *i = 0.0;
        }
        let mut slice = res.index_mut((ROW * DIM.., 0..ROW * DIM));
        for i in slice.iter_mut() {
            *i = 0.0;
        }
        (Some(CsrMatrix::from(&res)), self.beam.n_prims)
    }

    fn hessian_inverse_mut<'a>(&'a mut self, _x: &DVector<f64>) -> &'a CscMatrix<f64> {
        todo!()
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
        let mut p = plane(ROW, COL, Some(SIZE), Some(SIZE), None);

        // init velocity
        for i in 0..COL {
            for j in 0..ROW {
                p.velos[DIM * (i * ROW + j)] =
                    -1.0 * (i as f64) * (i as f64) * (i as f64 / 20.0) * SIZE * SIZE * SIZE;
            }
        }

        let mut g_vec = DVector::zeros(DIM * p.n_verts);
        for i in ROW..p.n_verts {
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

    let solver = NewtonSolverMut {
        max_iter: 500,
        epi: 1e-3,
    };
    let linearsolver = NewtonCG::<JacobianPre<CsrMatrix<f64>>>::new();
    // let linearsolver = CscCholeskySolver {};
    let linesearch = SimpleLineSearch {
        alpha: 0.9,
        tol: 0.01,
        epi: 1e-7,
    };
    let mut b = Scenario::new(
        problem,
        solver,
        linearsolver,
        linesearch,
        #[cfg(feature = "log")]
        format!("output/log/{FILENAME}_E_{E}_NU_{NU}_ROW_{ROW}_DENSITY_{DENSITY}_COL_{COL}_SIZE_{SIZE}/"),
        #[cfg(feature = "log")]
        format!("ACTIVESETEPI_0_NEIGH_0_.txt"),
        #[cfg(feature = "log")]
        format!("{COMMENT}\nE: {E}\nNU: {NU}\nROW: {ROW}\nCOL: {COL}\nDENSITY: {DENSITY}\nSIZE: {SIZE}\nACTIVE_SET_EPI: {ACTIVE_SET_EPI}\nNEIGH: {NEIGHBOR_LEVEL}"),
    );

    #[cfg(not(feature = "log"))]
    let start = std::time::Instant::now();
    for _i in 0..TOTAL_FRAME {
        println!("running frame: {}", _i);
        b.step();
    }
    #[cfg(not(feature = "log"))]
    println!("time spent {}", start.elapsed().as_secs_f32());
}
