use autodiff::Gradient;
use autodiff::Hessian;
use autodiff::MyScalar;
use nalgebra::DMatrix;
use nalgebra::DVector;
use nalgebra::SVector;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use crate::Energy;
use crate::Mesh2d;

impl Mesh2d {
    pub fn save_to_obj<P: AsRef<Path>>(&self, path: P) {
        let mut file = File::create(path).unwrap();
        writeln!(file, "g obj").unwrap();
        for i in 0..self.n_verts {
            writeln!(
                file,
                "v  {}  {}  {} ",
                self.verts[i * 2],
                self.verts[i * 2 + 1],
                0.0
            )
            .unwrap();
        }
        writeln!(file).unwrap();
        for inds in self.prim_connected_vert_indices.iter() {
            writeln!(
                file,
                "f  {}  {}  {} ",
                inds[0] + 1,
                inds[1] + 1,
                inds[2] + 1
            )
            .unwrap();
        }
    }

    pub fn prim_energy<E: Energy<6, 2>, T>(
        &self,
        index: usize,
        energy: &E,
        vert_vec: SVector<f64, 6>,
    ) -> T
    where
        T: MyScalar,
    {
        let vert_gradient_vec = T::as_myscalar_vec(vert_vec);
        let inv_mat = self.ma_invs[index];
        let inv_mat = T::as_constant_mat(inv_mat);
        let square = self.volumes[index];
        let ene = energy.energy(vert_gradient_vec, inv_mat, square);
        ene
    }

    pub fn get_indices(&self, i: usize) -> [usize; 6] {
        let ind = self.prim_connected_vert_indices[i];
        let indices = [
            ind[0] * 2,
            ind[0] * 2 + 1,
            ind[1] * 2,
            ind[1] * 2 + 1,
            ind[2] * 2,
            ind[2] * 2 + 1,
        ];
        indices
    }

    pub fn elastic_apply<E: Energy<6, 2>>(&self, x: &DVector<f64>, energy: &E) -> f64 {
        let mut res = 0.0;
        for i in 0..self.n_prims {
            let indices = self.get_indices(i);
            let mut vert_vec = SVector::<f64, 6>::zeros();
            vert_vec
                .iter_mut()
                .zip(indices.iter())
                .for_each(|(g_i, i)| *g_i = x[*i]);
            let energy: f64 = self.prim_energy(i, energy, vert_vec);
            res += energy;
        }
        res
    }
    pub fn elastic_gradient<E: Energy<6, 2>>(&self, x: &DVector<f64>, energy: &E) -> DVector<f64> {
        let mut res = DVector::zeros(x.len());
        for i in 0..self.n_prims {
            let indices = self.get_indices(i);
            let mut vert_vec = SVector::<f64, 6>::zeros();
            vert_vec
                .iter_mut()
                .zip(indices.iter())
                .for_each(|(g_i, i)| *g_i = x[*i]);
            let energy: Gradient<6> = self.prim_energy(i, energy, vert_vec);
            let grad = energy.gradient();
            indices
                .iter()
                .zip(grad.iter())
                .for_each(|(i_i, g_i)| res[*i_i] += g_i);
        }
        res
    }

    pub fn elastic_hessian<E: Energy<6, 2>>(&self, x: &DVector<f64>, energy: &E) -> DMatrix<f64> {
        let mut res = DMatrix::zeros(x.len(), x.len());
        for i in 0..self.n_prims {
            let indices = self.get_indices(i);
            let mut vert_vec = SVector::<f64, 6>::zeros();
            vert_vec
                .iter_mut()
                .zip(indices.iter())
                .for_each(|(g_i, i)| *g_i = x[*i]);
            let energy: Hessian<6> = self.prim_energy(i, energy, vert_vec);
            let small_hessian = energy.hessian();
            for i in 0..6 {
                for j in 0..6 {
                    res[(indices[i], indices[j])] += small_hessian[(i, j)];
                }
            }
        }
        res
    }
}
