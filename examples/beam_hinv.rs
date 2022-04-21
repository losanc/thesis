use mesh::*;
use nalgebra::{DMatrix, DVector, SMatrix, SVector};
use nalgebra_sparse::{factorization::CscCholesky, CscMatrix, CsrMatrix};
use optimization::*;
use thesis::scenarios::{Scenario, ScenarioProblem};
mod parameters2d;
use parameters2d::*;

const FILENAME: &'static str = "beaminv.txt";
const COMMENT: &'static str = "inverse hessian";

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
    l_matrix: CscMatrix<f64>,
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
    fn hessian_inverse_mut<'a>(&'a mut self, x: &DVector<f64>) -> &'a CscMatrix<f64> {
        let update_triangle_list = self
            .active_set
            .iter()
            .map(|x| self.beam.vert_connected_prim_indices[*x].clone())
            .flatten()
            .collect::<std::collections::HashSet<usize>>();

        let l = &mut self.l_matrix;
        // println!("{}  {}",update_triangle_list.len(),self.beam.n_prims);

        for i in update_triangle_list {
            let indices = self.beam.get_indices(i);

            let mut vert_vec = SVector::<f64, 6>::zeros();
            vert_vec
                .iter_mut()
                .zip(indices.iter())
                .for_each(|(g_i, i)| *g_i = x[*i]);

            let small_hessian = self.beam.prim_projected_hessian(i, &self.energy, vert_vec);
            let diff = small_hessian - self.hessian_list[i];

            let mut eigendecomposition = diff.symmetric_eigen();

            for j in 0..CO_NUM {
                let eigen_value = eigendecomposition.eigenvalues[j];
                if eigen_value.abs() < 1e-5 {
                    continue;
                }

                let update_flag;
                let mut vector = eigendecomposition.eigenvectors.column_mut(j);

                if eigen_value > 0.0 {
                    update_flag = 1.0;
                    vector *= eigen_value.sqrt();
                } else {
                    update_flag = -1.0;
                    vector *= (-eigen_value).sqrt();
                }
                let (col_offsets, row_indices, values) = l.csc_data_mut();
                let n = col_offsets.len() - 1;

                let mut x_vec = DVector::<f64>::zeros(n);
                x_vec[indices[0]] = vector[0];
                x_vec[indices[1]] = vector[1];
                x_vec[indices[2]] = vector[2];
                x_vec[indices[3]] = vector[3];
                x_vec[indices[4]] = vector[4];
                x_vec[indices[5]] = vector[5];

                x_vec.index_mut((0..DIM * NFIXED_VERT, 0)).fill(0.0);

                unsafe {
                    for k in 0..n {
                        let xk = x_vec.get_unchecked(k);
                        if xk.abs() < 1e-6 {
                            continue;
                        }
                        let lkk = values.get_unchecked_mut(*col_offsets.get_unchecked(k));
                        let lkkv = *lkk;
                        let r = lkkv*lkkv +update_flag*xk*xk;
                        assert!(r > 0.0);
                        let r = r.sqrt();
                        let other_c = r / lkkv;
                        let s = xk / lkkv;
                        *lkk = r;

                        // possible for simd optimization
                        for m in col_offsets.get_unchecked(k) + 1..*col_offsets.get_unchecked(k + 1)
                        {
                            let r_index = *row_indices.get_unchecked(m);
                            let v = values.get_unchecked_mut(m);
                            let x_r_index = x_vec.get_unchecked_mut(r_index);
                            *v = (*v + update_flag* s * *x_r_index) / other_c;
                            *x_r_index *= other_c;
                            *x_r_index -= s * *v;
                        }
                    }
                }
            }
            self.hessian_list[i] = small_hessian;
        }
        &self.l_matrix
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
        let mut p = plane(NFIXED_VERT, c, Some(SIZE), Some(SIZE), None);

        // init velocity
        for i in 0..c {
            for j in 0..NFIXED_VERT {
                p.velos[DIM * (i * NFIXED_VERT + j)] =
                    -1.0 * (i as f64) * (i as f64) * (i as f64 / 20.0) * SIZE * SIZE * SIZE;
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
        let elastic_psd_hessian = p.elastic_hessian_projected(&p.verts, &energy);

        let mut init_hessian = CscMatrix::from(&elastic_psd_hessian) + CscMatrix::from(&mass);
        for (i, j, k) in init_hessian.triplet_iter_mut() {
            if (i < DIM * NFIXED_VERT || j < DIM * NFIXED_VERT) && (i != j) {
                *k = 0.0;
            }
        }

        let l_matrix = CscCholesky::factor(&init_hessian).unwrap().l().clone();

        let mut old_hessian_list = Vec::<SMatrix<f64, CO_NUM, CO_NUM>>::new();
        for i in 0..p.n_prims {
            let mut vert_vec = SVector::<f64, CO_NUM>::zeros();
            let indices = p.get_indices(i);

            vert_vec
                .iter_mut()
                .zip(indices.iter())
                .for_each(|(g_i, i)| *g_i = p.verts[*i]);

            let small_hessian = p.prim_projected_hessian(i, &energy, vert_vec);
            old_hessian_list.push(small_hessian);
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
            l_matrix,
        };
        scenario
    }
}

fn main() {
    let problem = BeamScenario::new("beam_hinv");

    let solver = NewtonInverseSolver {
        max_iter: 500,
        epi: 1e-3,
    };
    let mut linearsolver = NewtonCG::<JacobianPre<CsrMatrix<f64>>>::new();
    linearsolver.tol = 1e-10;
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
