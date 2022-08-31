#![allow(non_snake_case)]
#![allow(unused_variables)]
use mesh::*;
use nalgebra::{DMatrix, DVector, SMatrix, SVector};
use nalgebra_sparse::{CscMatrix, CsrMatrix};
use optimization::*;
use thesis::scenarios::{Scenario, ScenarioProblem};

pub const DIM: usize = 2;
pub const CO_NUM: usize = DIM * (DIM + 1);
pub type EnergyType = NeoHookean2d;

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
    init_hessian: DMatrix<f64>,
    frame: usize,
    damp: f64,
    neighbor_level: usize,
    active_set_epi: f64,
    n_fixed: usize,
    modification: HessianModification,
    first_flag: bool,
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

        let mut slice = res.index_mut((0..self.n_fixed * DIM, 0));
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
            let boundary = self.active_set_epi * res.amax();
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
        res += self
            .beam
            .elastic_hessian(x, &self.energy, HessianModification::NoModification);

        let mut slice = res.index_mut((0..self.n_fixed * DIM, self.n_fixed * DIM..));
        for i in slice.iter_mut() {
            *i = 0.0;
        }
        let mut slice = res.index_mut((self.n_fixed * DIM.., 0..self.n_fixed * DIM));
        for i in slice.iter_mut() {
            *i = 0.0;
        }
        Some(CsrMatrix::from(&res))
    }

    fn hessian_mut(&mut self, x: &DVector<f64>) -> (Option<Self::HessianType>, usize) {
        
        // dense version of hessian matrix
        let mut res = DMatrix::<f64>::zeros(x.len(), x.len());
        res = res + self.inertia_hessian(x);

        if self.first_flag{
            res +=&self.init_hessian;
            self.first_flag = false;
            return  (Some(CsrMatrix::from(&res)), 0);

        }

        let mut update_triangle_list = self
            .active_set
            .iter()
            .map(|x| self.beam.vert_connected_prim_indices[*x].clone())
            .flatten()
            .collect::<std::collections::HashSet<usize>>();

        for _ in 0..self.neighbor_level {
            update_triangle_list = update_triangle_list
                .iter()
                .map(|x| self.beam.prim_connected_vert_indices[*x].clone())
                .flatten()
                .map(|x| self.beam.vert_connected_prim_indices[x].clone())
                .flatten()
                .collect::<std::collections::HashSet<usize>>();
        }
        let mut update_triangle_list = update_triangle_list.into_iter().collect::<Vec<_>>();
        update_triangle_list.sort();
        let assem_count = update_triangle_list.len();

        for i in update_triangle_list {
            let mut vert_vec = SVector::<f64, CO_NUM>::zeros();
            let indices = self.beam.get_indices(i);

            vert_vec
                .iter_mut()
                .zip(indices.iter())
                .for_each(|(g_i, i)| *g_i = x[*i]);

            let energy_hessian =
                self.beam
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

        res += &self.init_hessian;

        let mut slice = res.index_mut((0..self.n_fixed * DIM, self.n_fixed * DIM..));
        for i in slice.iter_mut() {
            *i = 0.0;
        }
        let mut slice = res.index_mut((self.n_fixed * DIM.., 0..self.n_fixed * DIM));
        for i in slice.iter_mut() {
            *i = 0.0;
        }
        (Some(CsrMatrix::from(&res)), assem_count)
    }

    fn hessian_inverse_mut<'a>(&'a mut self, _x: &DVector<f64>) -> &'a CscMatrix<f64> {
        todo!()
    }
}

impl ScenarioProblem for BeamScenario {
    #[inline]
    fn initial_guess(&self) -> DVector<f64> {
        self.beam.verts.clone()
    }
    #[inline]
    fn set_all_vertices_vector(&mut self, vertices: DVector<f64>) {
        let velocity = self.damp * ((&vertices - &self.beam.verts) / self.dt);
        self.beam.velos = velocity;
        self.beam.verts = vertices;
    }
    #[inline]
    fn save_to_file(&self, frame: usize) {
        self.beam
            .save_to_obj(format!("output/mesh/{}{}.obj", self.name, frame));
    }
    #[inline]
    fn frame_init(&mut self) {
        self.x_tao = &self.beam.verts + self.dt * &self.beam.velos;
    }
    #[inline]
    fn frame_end(&mut self) {
        self.frame += 1;
        self.first_flag = true;
    }
}

impl BeamScenario {
    pub fn new(name: &str) -> Self {
        let args: Vec<String> = std::env::args().collect();
        let E = args[1].parse::<f64>().unwrap();
        let NU = args[2].parse::<f64>().unwrap();
        let MIU = E / (2.0 * (1.0 + NU));
        let LAMBDA = (E * NU) / ((1.0 + NU) * (1.0 - 2.0 * NU));
        let DT = args[3].parse::<f64>().unwrap();
        let DENSITY = args[4].parse::<f64>().unwrap();
        let ROW = args[5].parse::<usize>().unwrap();
        let COL = args[6].parse::<usize>().unwrap();
        let DAMP = args[7].parse::<f64>().unwrap();
        let SIZE = args[8].parse::<f64>().unwrap();

        let ACTIVE_SET_EPI = args[9].parse::<f64>().unwrap();
        let NEIGHBOR_LEVEL = args[10].parse::<usize>().unwrap();
        let FILENAME = &args[11];
        let MODIFICATION = &args[14];
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

        let mut p = plane(
            ROW,
            COL,
            Some(SIZE),
            Some(SIZE),
            Some(DENSITY),
            uniform,
            seed,
        );

        // for i in 0..ROW {
        //     p.verts[DIM * i + 1] -= 1.0;
        // }
        // // for i in 0..ROW {
        // //     p.verts[DIM * (i+ROW) + 1] -= 0.5;
        // // }

        // init velocity
        for i in 0..COL {
            let x = p.verts[DIM * (i * ROW) + 1];
            for j in 0..ROW {
                p.velos[DIM * (i * ROW + j)] = -5.0 * x * x * x * SIZE * SIZE * SIZE;
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

        let scenario = Self {
            dt: DT,
            energy,
            name: String::from(FILENAME),
            mass,
            beam: p,
            x_tao: DVector::<f64>::zeros(1),
            g_vec,
            active_set: std::collections::HashSet::<usize>::new(),
            hessian_list: old_hessian_list,
            init_hessian,
            frame: 0,
            damp: DAMP,
            neighbor_level: NEIGHBOR_LEVEL,
            active_set_epi: ACTIVE_SET_EPI,
            n_fixed: ROW,
            modification: modi,
            first_flag:true,
        };
        scenario
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
    let ROW = args[5].parse::<usize>().unwrap();
    let COL = args[6].parse::<usize>().unwrap();
    // let DAMP = args[7].parse::<f64>().unwrap();
    let SIZE = args[8].parse::<f64>().unwrap();

    let ACTIVE_SET_EPI = args[9].parse::<f64>().unwrap();
    let NEIGHBOR_LEVEL = args[10].parse::<usize>().unwrap();

    let FILENAME = &args[11];
    let COMMENT = &args[12];
    let TOTAL_FRAME = args[13].parse::<usize>().unwrap();
    let MODIFICATION = &args[14];

    let precision = args[15].parse::<usize>().unwrap();

    let problem = BeamScenario::new("beamnew");

    let solver = NewtonSolverMut {
        max_iter: 1000,
        epi: 1e-3,
    };
    // let linearsolver = CscCholeskySolver {};
    // let linearsolver = NewtonCG::<JacobianPre<CsrMatrix<f64>>>::new();
    let mut linearsolver = MINRESLinear::<CsrMatrix<f64>>::new();
    linearsolver.epi = 1e-7;
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
        format!("output/log/{FILENAME}_E_{:e}_NU_{NU}_ROW_{ROW}_DENSITY_{DENSITY}_COL_{COL}_SIZE_{SIZE}_DT_{:.3}/",E,DT),
        #[cfg(feature = "log")]
        format!("ACTIVESETEPI_{:.precision$}_NEIGH_{:02}_.txt",ACTIVE_SET_EPI,NEIGHBOR_LEVEL),
        #[cfg(feature = "log")]
        format!("{COMMENT}\nE: {E}\nNU: {NU}\nROW: {ROW}\nCOL: {COL}\nDENSITY: {DENSITY}\nSIZE: {SIZE}\nACTIVE_SET_EPI: {ACTIVE_SET_EPI}\nNEIGH: {NEIGHBOR_LEVEL}\nMODIFICATION: {MODIFICATION}\n"),
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
