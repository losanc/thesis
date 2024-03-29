#![allow(non_snake_case)]
#![allow(unused_variables)]
use std::f64::consts::PI;

use mesh::Mesh3d;
use mesh::{cube, HessianModification};
use nalgebra::{DMatrix, DVector, SMatrix, SVector};
use nalgebra_sparse::*;
use optimization::*;
use thesis::scenarios::{Scenario, ScenarioProblem};

pub const DIM: usize = 3;
pub const CO_NUM: usize = 12;
type EnergyType = mesh::NeoHookean3d;

pub struct BouncingUpdateScenario {
    cube: Mesh3d,
    dt: f64,
    name: String,
    energy: EnergyType,
    x_tao: DVector<f64>,
    g_vec: DVector<f64>,
    mass: CsrMatrix<f64>,

    active_set: std::collections::HashSet<usize>,
    hessian_list: Vec<SMatrix<f64, CO_NUM, CO_NUM>>,
    init_hessian: DMatrix<f64>,
    active_set_epi: f64,
    neighbor_level: usize,
    damp: f64,
    modification: HessianModification,
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
        res += self.cube.elastic_apply(x, &self.energy);
        res
    }

    fn gradient(&self, x: &DVector<f64>) -> Option<DVector<f64>> {
        let mut res = DVector::<f64>::zeros(x.len());
        res += self.inertia_gradient(x);

        res += self.cube.elastic_gradient(x, &self.energy);

        Some(res)
    }

    fn gradient_mut(&mut self, x: &DVector<f64>) -> Option<DVector<f64>> {
        let res = self.gradient(x).unwrap();

        // reset active set
        {
            self.active_set.clear();
            let boundary = self.active_set_epi * res.amax();
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
            .map(|x| self.cube.vert_connected_prim_indices[*x].clone())
            .flatten()
            .collect::<std::collections::HashSet<usize>>();

        for _ in 0..self.neighbor_level {
            update_triangle_list = update_triangle_list
                .iter()
                .map(|x| self.cube.prim_connected_vert_indices[*x].clone())
                .flatten()
                .map(|x| self.cube.vert_connected_prim_indices[x].clone())
                .flatten()
                .collect::<std::collections::HashSet<usize>>();
        }
        let mut update_triangle_list = update_triangle_list.into_iter().collect::<Vec<_>>();
        update_triangle_list.sort();

        for i in &update_triangle_list {
            let i = *i;
            let mut vert_vec = SVector::<f64, CO_NUM>::zeros();
            let indices = self.cube.get_indices(i);

            vert_vec
                .iter_mut()
                .zip(indices.iter())
                .for_each(|(g_i, i)| *g_i = x[*i]);
            let energy_hessian =
                self.cube
                    .prim_projected_hessian(i, &self.energy, vert_vec, self.modification);

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
        let sparse = CsrMatrix::from(&self.init_hessian);

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
        self.cube.verts.clone()
    }
    fn set_all_vertices_vector(&mut self, vertices: DVector<f64>) {
        let velocity = self.damp * ((&vertices - &self.cube.verts) / self.dt);
        self.cube.velos = velocity;
        self.cube.verts = vertices;
    }
    fn save_to_file(&self, frame: usize) {
        self.cube
            .save_to_obj(format!("output/mesh/{}{}.vtk", self.name, frame));
    }
    fn frame_init(&mut self) {
        self.x_tao = &self.cube.verts + self.dt * &self.cube.velos;
    }
    fn frame_end(&mut self) {
        // self.frame+=1;
    }
}

impl BouncingUpdateScenario {
    pub fn new(name: &str) -> Self {
        let args: Vec<String> = std::env::args().collect();

        let E = args[1].parse::<f64>().unwrap();
        let NU = args[2].parse::<f64>().unwrap();
        let MIU = E / (2.0 * (1.0 + NU));
        let LAMBDA = (E * NU) / ((1.0 + NU) * (1.0 - 2.0 * NU));
        let DT = args[3].parse::<f64>().unwrap();
        let DENSITY = args[4].parse::<f64>().unwrap();
        let DAMP = args[5].parse::<f64>().unwrap();

        let ACTIVE_SET_EPI = args[6].parse::<f64>().unwrap();
        let NEIGHBOR_LEVEL = args[7].parse::<usize>().unwrap();

        let FILENAME = &args[8];
        let COMMENT = &args[9];
        let TOTAL_FRAME = args[10].parse::<usize>().unwrap();
        let MODIFICATION = &args[11];

        let ROW = args[13].parse::<usize>().unwrap();
        let COLUMN = args[14].parse::<usize>().unwrap();
        let STACK = args[15].parse::<usize>().unwrap();
        let uniform = args[16].parse::<bool>().unwrap();
        let seed = args[17].parse::<u64>().unwrap();

        let modi: HessianModification;
        match MODIFICATION.as_str() {
            "no" => {
                modi = HessianModification::NoModification;
            }
            "flip" => {
                modi = HessianModification::FlipMinusEigenvalues;
            }
            "remove" => {
                modi = HessianModification::RemoveMinusEigenvalues;
            }
            _ => {
                panic!("unknown ");
            }
        }

        let mut p = cube(
            ROW,
            COLUMN,
            STACK,
            None,
            None,
            None,
            Some(DENSITY),
            uniform,
            seed,
        );

        let center_x = (ROW / 2) as f64;
        let center_y = (COLUMN / 2) as f64;
        for i in 0..p.n_verts {
            p.verts[DIM * i] -= center_x;
            p.verts[DIM * i + 1] -= center_y;

            let length = (p.verts[DIM * i] * p.verts[DIM * i]
                + p.verts[DIM * i + 1] * p.verts[DIM * i + 1])
                .sqrt();
            if length < 1e-5 {
                p.verts[DIM * i + 2] *= 0.5;
                continue;
            }

            let mut angle = (p.verts[DIM * i] / length).asin();
            let cosvalue = p.verts[DIM * i + 1] / length;
            if cosvalue < 0.0 {
                angle = PI - angle;
            }
            angle += p.verts[DIM * i + 2] / 5.0;
            p.verts[DIM * i] = angle.sin() * length;
            p.verts[DIM * i + 1] = angle.cos() * length;
            p.verts[DIM * i + 2] *= 0.5;
        }

        let g_vec = DVector::zeros(DIM * p.n_verts);
        let energy = EnergyType {
            mu: MIU,
            lambda: LAMBDA,
        };

        let mass = DMatrix::from_diagonal(&p.masss) / (DT * DT);

        let init_hessian = p.elastic_hessian(&p.verts, &energy, modi);

        let mut old_hessian_list = Vec::<SMatrix<f64, CO_NUM, CO_NUM>>::new();
        for i in 0..p.n_prims {
            let mut vert_vec = SVector::<f64, CO_NUM>::zeros();
            let indices = p.get_indices(i);

            vert_vec
                .iter_mut()
                .zip(indices.iter())
                .for_each(|(g_i, i)| *g_i = p.verts[*i]);
            let energy_hessian = p.prim_projected_hessian(i, &energy, vert_vec, modi);
            old_hessian_list.push(energy_hessian);
        }

        Self {
            dt: DT,
            energy,
            name: String::from(name),
            mass: CsrMatrix::from(&mass),
            cube: p,

            x_tao: DVector::<f64>::zeros(1),
            g_vec,

            init_hessian,
            active_set: std::collections::HashSet::<usize>::new(),
            hessian_list: old_hessian_list,
            damp: DAMP,
            active_set_epi: ACTIVE_SET_EPI,
            neighbor_level: NEIGHBOR_LEVEL,
            modification: modi,
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let E = args[1].parse::<f64>().unwrap();
    let NU = args[2].parse::<f64>().unwrap();
    // let MIU = E / (2.0 * (1.0 + NU));
    // let LAMBDA = (E * NU) / ((1.0 + NU) * (1.0 - 2.0 * NU));
    let DT = args[3].parse::<f64>().unwrap();
    let DENSITY = args[4].parse::<f64>().unwrap();
    // let DAMP = args[5].parse::<f64>().unwrap();

    let ACTIVE_SET_EPI = args[6].parse::<f64>().unwrap();
    let NEIGHBOR_LEVEL = args[7].parse::<usize>().unwrap();

    let FILENAME = &args[8];
    let COMMENT = &args[9];
    let TOTAL_FRAME = args[10].parse::<usize>().unwrap();
    let MODIFICATION = &args[11];

    let precision = args[12].parse::<usize>().unwrap();

    let problem = BouncingUpdateScenario::new(FILENAME);

    let solver = NewtonSolverMut {
        max_iter: 300,
        epi: 1e-1,
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
        format!("output/log/{FILENAME}_E_{:e}_NU_{NU}_DENSITY_{DENSITY}_DT_{:.3}/",E,DT),
        #[cfg(feature = "log")]
        format!("ACTIVESETEPI_{:.precision$}_NEIGH_{:02}_.txt", ACTIVE_SET_EPI,NEIGHBOR_LEVEL),
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
