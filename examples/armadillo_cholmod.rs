#![allow(dead_code)]
#![allow(unused_imports)]
use std::mem::MaybeUninit;

use mesh::*;
use nalgebra::{dvector, DMatrix, DVector, Dynamic, SMatrix, SVector, U3};
use nalgebra_sparse::{factorization::CscCholesky, CscMatrix, CsrMatrix};
use optimization::*;
use thesis::{
    bindings::*,
    cholmod_common,
    csc::{csc_convert, Csc64},
    scenarios::{Scenario, ScenarioProblem},
};
mod armadillopara;
use armadillopara::*;

const FILENAME: &'static str = "armadillo_cholmod.txt";
const COMMENT: &'static str = "armadillo cholmod";

pub struct ArmadilloScenario {
    armadillo: Mesh3d,
    dt: f64,
    name: String,
    energy: EnergyType,
    x_tao: DVector<f64>,
    g_vec: DVector<f64>,
    mass: DMatrix<f64>,
    hessian_list: Vec<SMatrix<f64, CO_NUM, CO_NUM>>,
    l: *mut thesis::cholmod_factor,
    c: thesis::cholmod_common,
    active_set: std::collections::HashSet<usize>,
}

impl Problem for ArmadilloScenario {
    type HessianType = DMatrix<f64>;

    fn apply(&self, x: &DVector<f64>) -> f64 {
        self.my_apply(x)
    }

    fn gradient(&self, x: &DVector<f64>) -> Option<DVector<f64>> {
        self.my_gradient(x)
    }

    fn hessian_inverse_mut<'a>(
        &'a mut self,
        _x: &DVector<f64>,
        // ) -> nalgebra_sparse::factorization::CscCholesky<f64>;
    ) -> &'a CscMatrix<f64> {
        todo!()
    }
}

impl ArmadilloScenario {
    fn inertia_apply(&self, x: &DVector<f64>) -> f64 {
        let temp = x - &self.x_tao - &self.g_vec * (self.dt * self.dt);
        let res = temp.dot(&(&self.mass * &temp));
        res / 2.0
    }

    fn inertia_gradient(&self, x: &DVector<f64>) -> DVector<f64> {
        let res_grad = &self.mass * (x - &self.x_tao - &self.g_vec * (self.dt * self.dt));
        res_grad
    }

    fn my_apply(&self, x: &DVector<f64>) -> f64 {
        let mut res = 0.0;
        res += self.inertia_apply(x);
        res += self.armadillo.elastic_apply(x, &self.energy);
        res
    }
    fn inertia_hessian<'a>(&'a self, _x: &DVector<f64>) -> &'a DMatrix<f64> {
        // self.mass has already been divided by dt*dt when constructing it
        &self.mass
    }

    fn my_gradient_mut(&mut self, x: &DVector<f64>) -> Option<DVector<f64>> {
        let res = self.my_gradient(x).unwrap();
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

    fn my_gradient(&self, x: &DVector<f64>) -> Option<DVector<f64>> {
        let mut res = DVector::<f64>::zeros(x.len());
        res += self.inertia_gradient(x);
        res += self.armadillo.elastic_gradient(x, &self.energy);

        let mut slice = res.index_mut((0..NFIXED_VERT * DIM, 0));
        for i in slice.iter_mut() {
            *i = 0.0;
        }

        Some(res)
    }

    fn my_hessian(&self, x: &DVector<f64>) -> DMatrix<f64> {
        let mut res = DMatrix::<f64>::zeros(x.len(), x.len());
        res = res + self.inertia_hessian(x);
        res += self.armadillo.elastic_hessian_projected(x, &self.energy);

        let mut slice = res.index_mut((0..NFIXED_VERT * DIM, NFIXED_VERT * DIM..));
        for i in slice.iter_mut() {
            *i = 0.0;
        }
        let mut slice = res.index_mut((NFIXED_VERT * DIM.., 0..NFIXED_VERT * DIM));
        for i in slice.iter_mut() {
            *i = 0.0;
        }
        res
    }

    fn new_hessian(&mut self, x: &DVector<f64>) -> *mut cholmod_factor_struct {
        let update_triangle_list = self
            .active_set
            .iter()
            .map(|x| self.armadillo.vert_connected_prim_indices[*x].clone())
            .flatten()
            .collect::<std::collections::HashSet<usize>>();

        for i in update_triangle_list {
            let indices = self.armadillo.get_indices(i);

            let mut vert_vec = SVector::<f64, CO_NUM>::zeros();
            vert_vec
                .iter_mut()
                .zip(indices.iter())
                .for_each(|(g_i, i)| *g_i = x[*i]);

            let small_hessian = self
                .armadillo
                .prim_projected_hessian(i, &self.energy, vert_vec);
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
                    update_flag = 1;
                    vector *= eigen_value.sqrt();
                } else {
                    update_flag = 0;
                    vector *= (-eigen_value).sqrt();
                }

                let mut x_vec = DVector::<f64>::zeros(x.len());
                indices.iter().enumerate().for_each(|(i, x)| {
                    if *x >= DIM * NFIXED_VERT {
                        x_vec[*x] = vector[i]
                    }
                });

                unsafe {
                    let mut dense = thesis::csc::dense_convert(&mut x_vec, &mut self.c);
                    let mut sparse_vec = cholmod_dense_to_sparse(&mut dense, 1, &mut self.c);
                    let mut cnew = cholmod_submatrix(
                        sparse_vec,
                        (*self.l).Perm as *mut _,
                        (*self.l).n as i64,
                        std::ptr::null_mut(),
                        -1,
                        1,
                        1,
                        &mut self.c,
                    );
                    let res = cholmod_updown(update_flag, cnew, self.l, &mut self.c);
                    assert!(res == 1);
                    cholmod_free_sparse(&mut cnew, &mut self.c);
                    cholmod_free_sparse(&mut sparse_vec, &mut self.c);
                }
            }
            self.hessian_list[i] = small_hessian;
        }

        self.l
    }

    fn new() -> Self {
        let armadillo = armadillo();

        let energy = EnergyType {
            mu: MIU,
            lambda: LAMBDA,
        };
        let mut energy_hessian = armadillo.elastic_hessian_projected(&armadillo.verts, &energy);

        let mass = DMatrix::from_diagonal(&armadillo.masss) / (DT * DT);
        energy_hessian += &mass;

        let mut g_vec = DVector::zeros(DIM * armadillo.n_verts);
        for i in NFIXED_VERT..armadillo.n_verts {
            g_vec[DIM * i + 1] = -9.8;
        }

        let mut old_hessian_list = Vec::<SMatrix<f64, CO_NUM, CO_NUM>>::new();
        for i in 0..armadillo.n_prims {
            let mut vert_vec = SVector::<f64, CO_NUM>::zeros();
            let indices = armadillo.get_indices(i);

            vert_vec
                .iter_mut()
                .zip(indices.iter())
                .for_each(|(g_i, i)| *g_i = armadillo.verts[*i]);

            let small_hessian = armadillo.prim_projected_hessian(i, &energy, vert_vec);
            old_hessian_list.push(small_hessian);
        }

        let mut slice = energy_hessian.index_mut((0..NFIXED_VERT * DIM, NFIXED_VERT * DIM..));
        for i in slice.iter_mut() {
            *i = 0.0;
        }
        let mut slice = energy_hessian.index_mut((NFIXED_VERT * DIM.., 0..NFIXED_VERT * DIM));
        for i in slice.iter_mut() {
            *i = 0.0;
        }
        let mut csc = Csc64::from(&energy_hessian);

        unsafe {
            let mut cholmod_common: MaybeUninit<cholmod_common> = MaybeUninit::uninit();
            cholmod_start(cholmod_common.as_mut_ptr());

            let mut cholmod_sparse = csc_convert(&mut csc, cholmod_common.as_mut_ptr());
            let cholmod_analyze_ptr =
                cholmod_analyze(&mut cholmod_sparse, cholmod_common.as_mut_ptr());
            cholmod_factorize(
                &mut cholmod_sparse,
                cholmod_analyze_ptr,
                cholmod_common.as_mut_ptr(),
            );

            Self {
                armadillo,
                dt: DT,
                name: String::from("armadillo_cholmod"),
                energy,
                x_tao: DVector::<f64>::zeros(1),
                g_vec,
                mass,
                l: cholmod_analyze_ptr,
                c: cholmod_common.assume_init(),
                hessian_list: old_hessian_list,
                active_set: std::collections::HashSet::<usize>::new(),
            }
        }
    }

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
}

impl Drop for ArmadilloScenario {
    fn drop(&mut self) {
        unsafe {
            cholmod_free_factor(&mut self.l as *mut _, &mut self.c as *mut _);
            cholmod_finish(&mut self.c as *mut _);
        }
    }
}

fn main() {
    let mut p = ArmadilloScenario::new();
    let ls = SimpleLineSearch {
        alpha: 0.9,
        tol: 0.01,
        epi: 1e-7,
    };
    let start = std::time::Instant::now();
    for i in 0..TOTAL_FRAME {
        println!("frame {i}");
        unsafe {
            p.frame_init();
            let mut res = p.initial_guess();
            let mut g = p.my_gradient_mut(&res).unwrap();
            let mut h;
            let mut count = 0;
            while g.norm() > 1e-3 {
                // println!("gnorm {}",g.norm());
                count += 1;
                h = p.new_hessian(&res);
                let mut g_dense = thesis::csc::dense_convert(&mut g, &mut p.c);
                let mut cholmod_delta = cholmod_solve(CHOLMOD_A as i32, h, &mut g_dense, &mut p.c);

                let delta = thesis::csc::cholmod_dense_convert(cholmod_delta, &mut p.c);

                let scalar = ls.search(&p, &res, &delta);
                let delta = delta * scalar;
                res -= &delta;
                g = p.my_gradient_mut(&res).unwrap();
                cholmod_free_dense(&mut cholmod_delta, &mut p.c);
                if count > 300 {
                    break;
                }
            }
            p.set_all_vertices_vector(res);
            p.save_to_file(i);
            // println!("{count}");
        }
    }
    println!("{}", start.elapsed().as_secs_f32());
}
