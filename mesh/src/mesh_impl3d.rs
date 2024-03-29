use autodiff::Gradient;
use autodiff::Hessian;
use autodiff::MyScalar;
use nalgebra::DMatrix;
use nalgebra::DVector;
use nalgebra::SMatrix;
use nalgebra::SVector;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use crate::Energy;
use crate::HessianModification;
use crate::Mesh3d;
impl Mesh3d {
    pub fn save_to_obj<P: AsRef<Path>>(&self, path: P) {
        let mut file = File::create(path).unwrap();

        writeln!(file, "# vtk DataFile Version 4.2").unwrap();
        writeln!(file, "Cube example").unwrap();
        writeln!(file, "ASCII").unwrap();
        writeln!(file, "DATASET UNSTRUCTURED_GRID").unwrap();
        writeln!(file, "POINTS {} float", self.n_verts).unwrap();
        for i in 0..self.n_verts {
            writeln!(
                file,
                "{}  {}  {} ",
                self.verts[i * 3],
                self.verts[i * 3 + 1],
                self.verts[i * 3 + 2],
            )
            .unwrap();
        }
        writeln!(file, "CELLS {} {}", self.n_prims, 5 * self.n_prims).unwrap();
        for i in 0..self.n_prims {
            writeln!(
                file,
                "4  {}  {}  {} {} ",
                self.prim_connected_vert_indices[i][0],
                self.prim_connected_vert_indices[i][1],
                self.prim_connected_vert_indices[i][2],
                self.prim_connected_vert_indices[i][3],
            )
            .unwrap();
        }
        writeln!(file, "CELL_TYPES {}", self.n_prims).unwrap();
        for _ in 0..self.n_prims {
            writeln!(file, "10").unwrap();
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

    pub fn elastic_hessian<E: Energy<12, 3>>(
        &self,
        x: &DVector<f64>,
        energy: &E,
        modification: HessianModification,
    ) -> DMatrix<f64> {
        let mut res = DMatrix::zeros(x.len(), x.len());
        for i in 0..self.n_prims {
            let indices = self.get_indices(i);
            let mut vert_vec = SVector::<f64, 12>::zeros();
            vert_vec
                .iter_mut()
                .zip(indices.iter())
                .for_each(|(g_i, i)| *g_i = x[*i]);
            let energy: Hessian<12> = self.prim_energy(i, energy, vert_vec);
            let small_hessian;
            match modification {
                HessianModification::NoModification => small_hessian = energy.hessian(),
                HessianModification::RemoveMinusEigenvalues => {
                    let hessian = energy.hessian();
                    let mut eigendecom = hessian.symmetric_eigen();
                    for eigenvalue in eigendecom.eigenvalues.iter_mut() {
                        if *eigenvalue < 0.0 {
                            *eigenvalue = 0.0;
                        }
                    }
                    small_hessian = eigendecom.recompose();
                }
                HessianModification::FlipMinusEigenvalues => {
                    let hessian = energy.hessian();
                    let mut eigendecom = hessian.symmetric_eigen();
                    for eigenvalue in eigendecom.eigenvalues.iter_mut() {
                        if *eigenvalue < 0.0 {
                            *eigenvalue *= -1.0;
                        }
                    }
                    small_hessian = eigendecom.recompose();
                }
                HessianModification::InternalRemove => todo!(),
                HessianModification::InternalFlip => todo!(),
            }
            for i in 0..12 {
                for j in 0..12 {
                    res[(indices[i], indices[j])] += small_hessian[(i, j)];
                }
            }
        }
        res
    }

    pub fn prim_projected_hessian<E: Energy<12, 3>>(
        &self,
        i: usize,
        energy: &E,
        vert_vec: SVector<f64, 12>,
        modification: HessianModification,
    ) -> SMatrix<f64, 12, 12> {
        let energy: Hessian<12> = self.prim_energy(i, energy, vert_vec);
        match modification {
            HessianModification::NoModification => {
                return energy.hessian();
            }
            HessianModification::RemoveMinusEigenvalues => {
                let small_hessian = energy.hessian();

                let mut eigendecomposition = small_hessian.symmetric_eigen();
                for eigenvalue in eigendecomposition.eigenvalues.iter_mut() {
                    if *eigenvalue < 0.0 {
                        *eigenvalue = 0.0;
                    }
                }
                return eigendecomposition.recompose();
            }
            HessianModification::FlipMinusEigenvalues => {
                let small_hessian = energy.hessian();

                let mut eigendecomposition = small_hessian.symmetric_eigen();
                for eigenvalue in eigendecomposition.eigenvalues.iter_mut() {
                    if *eigenvalue < 0.0 {
                        *eigenvalue *= -1.0;
                    }
                }
                return eigendecomposition.recompose();
            }
            HessianModification::InternalRemove => todo!(),
            HessianModification::InternalFlip => todo!(),
        }
    }

    pub fn elastic_hessian_projected<E: Energy<12, 3>>(
        &self,
        x: &DVector<f64>,
        energy: &E,
    ) -> DMatrix<f64> {
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
            let mut eigendecomposition = small_hessian.symmetric_eigen();
            for eigenvalue in eigendecomposition.eigenvalues.iter_mut() {
                if *eigenvalue < 0.0 {
                    *eigenvalue = 0.0;
                }
            }
            let small_hessian = eigendecomposition.recompose();
            for i in 0..12 {
                for j in 0..12 {
                    res[(indices[i], indices[j])] += small_hessian[(i, j)];
                }
            }
        }
        res
    }
}
