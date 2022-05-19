use autodiff::Hessian;
use mesh::{armadillo, Mesh3d};
use nalgebra::{DMatrix, DVector, SMatrix, SVector};
use nalgebra_sparse::*;
use optimization::*;
use thesis::scenarios::{Scenario, ScenarioProblem};
mod armadillopara;
use armadillopara::*;
const FILENAME: &'static str = "armadillo_projected_fast.txt";
const COMMENT: &'static str = "naive projected newton";

pub struct BouncingUpdateScenario {
    armadillo: Mesh3d,
    dt: f64,
    name: String,
    energy: EnergyType,
    x_tao: DVector<f64>,
    g_vec: DVector<f64>,
    mass: CsrMatrix<f64>,

    active_set: std::collections::HashSet<usize>,
    hessian_list: Vec<SMatrix<f64, CO_NUM, CO_NUM>>,
    init_hessian: DMatrix<f64>,
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

    fn hessian_mut(&mut self, x: &DVector<f64>) -> (Option<Self::HessianType>, usize) {
        // dense version of hessian matrix
        let mut update_triangle_list = self
            .active_set
            .iter()
            .map(|x| self.armadillo.vert_connected_prim_indices[*x].clone())
            .flatten()
            .collect::<std::collections::HashSet<usize>>();

        for _ in 0..NEIGHBOR_LEVEL {
            update_triangle_list = update_triangle_list
                .iter()
                .map(|x| self.armadillo.prim_connected_vert_indices[*x].clone())
                .flatten()
                .map(|x| self.armadillo.vert_connected_prim_indices[x].clone())
                .flatten()
                .collect::<std::collections::HashSet<usize>>();
        }
        let mut update_triangle_list = update_triangle_list.into_iter().collect::<Vec<_>>();
        update_triangle_list.sort();

        for i in &update_triangle_list {
            let i = *i;
            let mut vert_vec = SVector::<f64, CO_NUM>::zeros();
            let indices = self.armadillo.get_indices(i);

            vert_vec
                .iter_mut()
                .zip(indices.iter())
                .for_each(|(g_i, i)| *g_i = x[*i]);
            let energy: Hessian<CO_NUM> = self.armadillo.prim_energy(i, &self.energy, vert_vec);
            let energy_hessian = energy.hessian();

            let mut eigendecomposition = energy_hessian.symmetric_eigen();
            for eigenvalue in eigendecomposition.eigenvalues.iter_mut() {
                if *eigenvalue < 0.0 {
                    *eigenvalue = -*eigenvalue;
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

        // res += &self.init_hessian;
        let mut sparse = CsrMatrix::from(&self.init_hessian);
        for (i, j, k) in sparse.triplet_iter_mut() {
            if i < NFIXED_VERT * DIM || j < NFIXED_VERT * DIM {
                *k = 0.0;
            }
        }

        (
            Some(self.inertia_hessian(x) + sparse),
            update_triangle_list.len(),
        )
    }

    fn hessian_inverse_mut<'a>(
        &'a mut self,
        _x: &DVector<f64>,
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
            .save_to_obj(format!("output/mesh/{}{}.obj", self.name, frame));
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
                    *eigenvalue = -*eigenvalue;
                }
            }
            let energy_hessian = eigendecomposition.recompose();
            old_hessian_list.push(energy_hessian);
        }

        Self {
            dt: DT,
            energy,
            name: String::from(name),
            mass: CsrMatrix::from(&mass),
            armadillo: p,

            x_tao: DVector::<f64>::zeros(1),
            g_vec,

            init_hessian,
            active_set: std::collections::HashSet::<usize>::new(),
            hessian_list: old_hessian_list,
        }
    }
}

fn main() {
    let problem = BouncingUpdateScenario::new("armadillonew");

    let solver = NewtonSolverMut {
        max_iter: 300,
        epi: 1e-3,
    };
    let linearsolver = NewtonCG::<JacobianPre<CsrMatrix<f64>>>::new();

    // let linearsolver = CscCholeskySolver {};
    let linesearch = SimpleLineSearch {
        alpha: 0.9,
        tol: 1e-5,
        epi: 0.0,
    };
    let mut b = Scenario::new(
        problem,
        solver,
        linearsolver,
        linesearch,
        #[cfg(feature = "log")]
        format!("output/log/{FILENAME}_E_{E}_NU_{NU}/"),
        #[cfg(feature = "log")]
        format!("ACTIVESETEPI_{ACTIVE_SET_EPI}_NEIGH_{NEIGHBOR_LEVEL}_.txt"),
        #[cfg(feature = "log")]
        format!("{COMMENT}\nE: {E}\nNU: {NU}\nACTIVE_SET_EPI: {ACTIVE_SET_EPI}\nNEIGH: {NEIGHBOR_LEVEL}")
        );

    let start = std::time::Instant::now();
    for _i in 0..TOTAL_FRAME {
        println!("{}", _i);
        b.step();
    }
    println!("{}", start.elapsed().as_secs_f32());
}
