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
use crate::Mesh3d;
impl Mesh3d {
    pub fn save_to_obj<P: AsRef<Path>>(&self, path: P) {
        let mut file = File::create(path).unwrap();
        writeln!(file, "g obj").unwrap();
        for i in 0..self.n_verts {
            writeln!(
                file,
                "v  {}  {}  {} ",
                self.verts[i * 3],
                self.verts[i * 3 + 1],
                self.verts[i * 3 + 2],
            )
            .unwrap();
        }
        writeln!(file).unwrap();
        for inds in self.surface.as_ref().unwrap().iter() {
            writeln!(
                file,
                "f  {}  {}  {} ",
                inds[0] + 1,
                inds[1] + 1,
                inds[2] + 1,
            )
            .unwrap();
        }
    }
    pub fn prim_energy<E: Energy<12, 3>, T>(
        &self,
        index: usize,
        energy: &E,
        vert_vec: SVector<f64, 12>,
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

    pub fn prim_energy_with_fixed<E: Energy<12, 3>, T>(
        &self,
        index: usize,
        energy: &E,
        vert_vec: SVector<f64, 12>,
        n_fixed: usize,
    ) -> T
    where
        T: MyScalar,
    {
        let mut vert_gradient_vec = T::as_myscalar_vec(vert_vec);
        let [v1, v2, v3, v4] = self.prim_connected_vert_indices[index];

        if v1 < n_fixed {
            vert_gradient_vec[0].to_consant();
            vert_gradient_vec[1].to_consant();
            vert_gradient_vec[2].to_consant();
        }
        if v2 < n_fixed {
            vert_gradient_vec[3].to_consant();
            vert_gradient_vec[4].to_consant();
            vert_gradient_vec[5].to_consant();
        }

        if v3 < n_fixed {
            vert_gradient_vec[6].to_consant();
            vert_gradient_vec[7].to_consant();
            vert_gradient_vec[8].to_consant();
        }

        if v4 < n_fixed {
            vert_gradient_vec[9].to_consant();
            vert_gradient_vec[10].to_consant();
            vert_gradient_vec[11].to_consant();
        }

        let inv_mat = self.ma_invs[index];
        let inv_mat = T::as_constant_mat(inv_mat);
        let square = self.volumes[index];
        let ene = energy.energy(vert_gradient_vec, inv_mat, square);
        ene
    }

    pub fn get_indices(&self, i: usize) -> [usize; 12] {
        let ind = self.prim_connected_vert_indices[i];
        let indices = [
            ind[0] * 3,
            ind[0] * 3 + 1,
            ind[0] * 3 + 2,
            ind[1] * 3,
            ind[1] * 3 + 1,
            ind[1] * 3 + 2,
            ind[2] * 3,
            ind[2] * 3 + 1,
            ind[2] * 3 + 2,
            ind[3] * 3,
            ind[3] * 3 + 1,
            ind[3] * 3 + 2,
        ];
        indices
    }

    pub fn elastic_apply<E: Energy<12, 3>>(&self, x: &DVector<f64>, energy: &E) -> f64 {
        let mut res = 0.0;
        for i in 0..self.n_prims {
            let indices = self.get_indices(i);
            let mut vert_vec = SVector::<f64, 12>::zeros();
            vert_vec
                .iter_mut()
                .zip(indices.iter())
                .for_each(|(g_i, i)| *g_i = x[*i]);
            let energy: f64 = self.prim_energy(i, energy, vert_vec);
            res += energy;
        }
        res
    }
    pub fn elastic_gradient<E: Energy<12, 3>>(&self, x: &DVector<f64>, energy: &E) -> DVector<f64> {
        let mut res = DVector::zeros(x.len());
        for i in 0..self.n_prims {
            let indices = self.get_indices(i);
            let mut vert_vec = SVector::<f64, 12>::zeros();
            vert_vec
                .iter_mut()
                .zip(indices.iter())
                .for_each(|(g_i, i)| *g_i = x[*i]);
            let energy: Gradient<12> = self.prim_energy(i, energy, vert_vec);
            let grad = energy.gradient();
            indices
                .iter()
                .zip(grad.iter())
                .for_each(|(i_i, g_i)| res[*i_i] += g_i);
        }
        res
    }

    pub fn elastic_hessian<E: Energy<12, 3>>(&self, x: &DVector<f64>, energy: &E) -> DMatrix<f64> {
        let mut res = DMatrix::zeros(x.len(), x.len());
        for i in 0..self.n_prims {
            let indices = self.get_indices(i);
            let mut vert_vec = SVector::<f64, 12>::zeros();
            vert_vec
                .iter_mut()
                .zip(indices.iter())
                .for_each(|(g_i, i)| *g_i = x[*i]);
            let energy: Hessian<12> = self.prim_energy(i, energy, vert_vec);
            let small_hessian = energy.hessian();
            for i in 0..12 {
                for j in 0..12 {
                    res[(indices[i], indices[j])] += small_hessian[(i, j)];
                }
            }
        }
        res
    }
}
