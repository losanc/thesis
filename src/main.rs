use matrixcompare::assert_matrix_eq;
use mesh::MeshType;
use mesh::Plane;
use na::{DMatrix, DVector, SMatrix, SVector};
use nalgebra as na;
use nalgebra_sparse as nas;
use nas::csr::CsrMatrix;
use nas::factorization::CscCholesky;
use nas::CooMatrix;
use nas::CscMatrix;

struct Scenario {
    p: Plane,
    g_vec: DVector<f64>,
    dt: f64,
    x_tao: DVector<f64>,
    x: DVector<f64>,
    mass: <Plane as MeshType>::MassMatrixType,
    // opt:
    frame: usize,
    tolerance: f64,
}

impl Scenario {
    fn new() -> Self {
        let p = Plane::new(4, 4);
        // TODO
        let mut g_vec = DVector::zeros(32);
        for i in 0..16 {
            g_vec[i * 2 + 1] = -9.8;
        }
        let x_tao = DVector::zeros(32);
        Self {
            g_vec: g_vec,
            dt: 0.01,
            frame: 0,
            x_tao: x_tao,
            mass: p.mass_matrix(),
            x: p.all_to_vert_vector(),
            p: p,
            tolerance: 1.0,
        }
    }
    fn energy(&self) -> f64 {
        let diff = &self.x - &self.x_tao;
        let ene = diff.transpose() * (&self.mass * (diff)) / (2.0 * self.dt * self.dt);
        let mut ene = ene[(0, 0)];
        for i in 0..self.p.n_trias {
            ene += self.p.primitive_elastic_energy(i);
        }
        ene
    }

    fn gradient(&self) -> na::DVector<f64> {
        // inertia and gravity gradient
        let mut res_grad = &self.mass * (&self.x - &self.x_tao) / (self.dt * self.dt);
        for i in 0..self.p.n_fixed {
            res_grad[i * 2] = 0.0;
            res_grad[i * 2 + 1] = 0.0;
        }

        // print!("first {}\n",res_grad);
        // elastic gradient
        for i in 0..self.p.n_trias {
            let vertex_index_3 = self.p.indices[i];
            let pri_ene_grad = self.p.primitive_elastic_energy_gradient(i).gradient();
            res_grad[2 * vertex_index_3[0]] += pri_ene_grad[0];
            res_grad[2 * vertex_index_3[0] + 1] += pri_ene_grad[1];
            res_grad[2 * vertex_index_3[1]] += pri_ene_grad[2];
            res_grad[2 * vertex_index_3[1] + 1] += pri_ene_grad[3];
            res_grad[2 * vertex_index_3[2]] += pri_ene_grad[4];
            res_grad[2 * vertex_index_3[2] + 1] += pri_ene_grad[5];
        }

        // print!("second {}\n",res_grad);
        res_grad
    }

    fn hessian(&self) -> CscMatrix<f64> {
        let mut res = &self.mass / (self.dt * self.dt);
        let mut big_hessian = CooMatrix::new(32, 32);
        for i in 0..self.p.n_trias {
            let vertex_index_3 = self.p.indices[i];
            let pri_ene_hessian = self.p.primitive_elastic_energy_hessian(i).hessian();
            for i in 0..3 {
                for j in 0..3 {
                    big_hessian.push(
                        2 * vertex_index_3[i],
                        2 * vertex_index_3[j],
                        pri_ene_hessian[(i * 2, j * 2)],
                    );
                    big_hessian.push(
                        2 * vertex_index_3[i],
                        2 * vertex_index_3[j] + 1,
                        pri_ene_hessian[(i * 2, j * 2 + 1)],
                    );
                    big_hessian.push(
                        2 * vertex_index_3[i] + 1,
                        2 * vertex_index_3[j],
                        pri_ene_hessian[(i * 2 + 1, j * 2)],
                    );
                    big_hessian.push(
                        2 * vertex_index_3[i] + 1,
                        2 * vertex_index_3[j] + 1,
                        pri_ene_hessian[(i * 2 + 1, j * 2 + 1)],
                    );
                }
            }
        }
        res + CscMatrix::from(&big_hessian)
    }

    fn step(&mut self) {
        self.frame += 1;

        self.x_tao = self.p.all_to_vert_vector()
            + self.dt * self.p.all_vels_to_vert_vector()
            + self.dt * self.dt * &self.g_vec;

        // self.
        let mut g = self.gradient();
        // print!("{}\n", g.norm());
        while g.norm() > self.tolerance {
            // for i in 0..10{
            let h = self.hessian();
            let delta = solver(&h, &g);
            // print!("res: {}\n", (&h*&delta-&g).norm());
            let product = delta.transpose() * &g;
            let product = product[(0, 0)];
            if product > 0.0 {
                self.x -= &delta;
            } else {
                self.x += &delta;
            }
            g = self.gradient();
            // print!("{}\n", g);
            // if g.norm()<self.tolerance{
            //     break;
            // }
        }

        print!("next frame \n\n");
        self.p
            .set_all_to_velo_vector(&((&self.x - self.p.all_to_vert_vector()) / self.dt));
        self.p.set_all_to_vert_vector(&self.x);

        self.p.save_to_obj(format!(
            "C:\\Users\\hui\\Desktop\\data\\example{}.obj",
            self.frame
        ));
    }
}

fn solver(A: &CscMatrix<f64>, b: &DVector<f64>) -> DMatrix<f64> {
    // let ma = CscMatrix::from(A);
    let res = CscCholesky::factor(A).unwrap().solve(b);
    res

    // let mut x = DVector::<f64>::zeros(b.nrows());
    // let mut r = b - A * &x;
    // if (r.norm() < 0.1) {
    //     return x;
    // }
    // let mut p = r.clone();
    // let mut alpha :f64;
    // let mut beta :f64;
    // let mut count = 0;
    // let mut numerate = r.norm();
    // for i in 0..10000 {

    //     let denominator = p.transpose() * (A * &p);
    //     let denominator = denominator[(0, 0)];
    //     if (denominator < 0.0) {
    //         if count == 0 {
    //             return b.clone();
    //         } else {
    //             return x;
    //         }
    //     }
    //     alpha = numerate / denominator;
    //     x += alpha * &p;
    //     r -= alpha * (A * &p);
    //     if (r.norm() < 0.1) {
    //         return x;
    //     }
    //     let new_num =r.norm();
    //     beta = new_num / numerate;

    //     p = &r + beta * &p;
    //     numerate = new_num;
    //     count += 1;
    // }
    // x
}

fn main() {
    let mut s = Scenario::new();
    for i in 0..500 {
        s.step();
    }
}

// fn test_grad() {
//     use autodiff as ad;
//     use na::SVector;
//     use nalgebra as na;
//     let mut t = SVector::<f64, 3>::zeros();
//     t[0] = 1.0;
//     t[1] = 2.0;
//     t[2] = 3.0;
//     let mut constant = SVector::<f64, 3>::zeros();
//     constant[0] = 1.0;
//     constant[1] = 2.0;
//     constant[2] = 3.0;
//     let res = ad::vector_to_gradients(&t);
//     let constant: SVector<ad::Gradient<3>, 3> = ad::constant_matrix_to_gradients(&constant);
//     let p = res.transpose() * (res + constant);
//     print!("{} \n\n\n", p);
// }

// fn test_hess() {
//     use autodiff as ad;
//     use na::SVector;
//     use nalgebra as na;
//     type VectorType = SVector<f64, 3>;
//     type HessianType = SVector<ad::Hessian<3>, 3>;
//     let mut t = VectorType::zeros();
//     t[0] = 1.0;
//     t[1] = 2.0;
//     t[2] = 3.0;
//     let mut constant = VectorType::zeros();
//     constant[0] = 1.0;
//     constant[1] = 2.0;
//     constant[2] = 3.0;
//     let res = ad::vector_to_hessians(&t);
//     let constant: HessianType = ad::constant_matrix_to_hessians(&constant);
//     let p = res.transpose() * (res + constant);
//     print!("{} ", p);
// }
