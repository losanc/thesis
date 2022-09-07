use autodiff::Gradient;
use autodiff::Hessian;
use autodiff::MyScalar;
use nalgebra::DMatrix;
use nalgebra::DVector;
use nalgebra::SMatrix;
use nalgebra::SVector;
use num::Zero;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use crate::Energy;
use crate::HessianModification;
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
        for i in 0..self.n_verts {
            writeln!(
                file,
                "vn  {}  {}  {} ",
                self.accls[i * 2],
                self.accls[i * 2 + 1],
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

    pub fn prim_projected_hessian<E: Energy<6, 2>>(
        &self,
        i: usize,
        energy: &E,
        vert_vec: SVector<f64, 6>,
        modification: HessianModification,
    ) -> SMatrix<f64, 6, 6> {
        let energy_value: Hessian<6> = self.prim_energy(i, energy, vert_vec);

        match modification {
            HessianModification::NoModification => {
                return energy_value.hessian();
            }
            HessianModification::RemoveMinusEigenvalues => {
                let small_hessian = energy_value.hessian();

                let mut eigendecomposition = small_hessian.symmetric_eigen();
                for eigenvalue in eigendecomposition.eigenvalues.iter_mut() {
                    if *eigenvalue < 0.0 {
                        *eigenvalue = 0.0;
                    }
                }
                return eigendecomposition.recompose();
            }
            HessianModification::FlipMinusEigenvalues => {
                let small_hessian = energy_value.hessian();

                let mut eigendecomposition = small_hessian.symmetric_eigen();
                for eigenvalue in eigendecomposition.eigenvalues.iter_mut() {
                    if *eigenvalue < 0.0 {
                        *eigenvalue *= -1.0;
                    }
                }
                return eigendecomposition.recompose();
            }
            HessianModification::InternalRemove => {
                let vert_gradient_vec = Hessian::<6>::as_myscalar_vec(vert_vec);
                    let inv_mat = self.ma_invs[i];
                    let inv_mat = Hessian::<6>::as_constant_mat(inv_mat);
                    let square = self.volumes[i];
                    let mat = nalgebra::matrix![
                        vert_gradient_vec[4]-vert_gradient_vec[0], vert_gradient_vec[2]-vert_gradient_vec[0];
                        vert_gradient_vec[5]-vert_gradient_vec[1], vert_gradient_vec[3]-vert_gradient_vec[1];
                    ];
                    let matrix_f = mat * inv_mat;
                    println!("{matrix_f}");
                    let mut partial_f_partial_x = SMatrix::<f64, 6, 4>::zeros();

                    partial_f_partial_x
                        .column_mut(0)
                        .copy_from(&matrix_f[(0, 0)].gradient);
                    partial_f_partial_x
                        .column_mut(1)
                        .copy_from(&matrix_f[(1, 0)].gradient);

                    partial_f_partial_x
                        .column_mut(2)
                        .copy_from(&matrix_f[(0, 1)].gradient);
                    partial_f_partial_x
                        .column_mut(3)
                        .copy_from(&matrix_f[(1, 1)].gradient);

                    let mut matrix_f_clone = nalgebra::matrix![
                        Hessian::<4>::zero(),Hessian::<4>::zero();
                        Hessian::<4>::zero(),Hessian::<4>::zero();
                    ];

                    matrix_f_clone[(0, 0)].value = matrix_f[(0, 0)].value;
                    matrix_f_clone[(1, 0)].value = matrix_f[(1, 0)].value;
                    matrix_f_clone[(0, 1)].value = matrix_f[(0, 1)].value;
                    matrix_f_clone[(1, 1)].value = matrix_f[(1, 1)].value;

                    matrix_f_clone[(0, 0)].gradient[0] = 1.0;
                    matrix_f_clone[(1, 0)].gradient[1] = 1.0;
                    matrix_f_clone[(0, 1)].gradient[2] = 1.0;
                    matrix_f_clone[(1, 1)].gradient[3] = 1.0;

                    let i1 = matrix_f_clone.transpose() * matrix_f_clone;
                    let i1 = i1.trace();
                    // let i2=  ?`
                    // Is i2 used in formula?

                    // i3 = matrix_f.determinate()
                    let i3 = matrix_f_clone[(0, 0)] * matrix_f_clone[(1, 1)]
                        - matrix_f_clone[(0, 1)] * matrix_f_clone[(1, 0)];

                    // invertion test
                    // assert!(i3 > 0.0)
                    let i3 = i3 * i3;

                    let logi3 = autodiff::MyLog::myln(i3);

                    let ene = (i1 - logi3 - 2.0) * (energy.mu() / 2.0)
                        + logi3 * logi3 * (energy.lambda() / 8.0);
                    let ene = ene * (<Hessian<4> as num::One>::one() * square);
                    let partial_phi_partial_f_square = ene.hessian();

                    let mut eigendecom = partial_phi_partial_f_square.symmetric_eigen();
                    for eigenvalue in eigendecom.eigenvalues.iter_mut() {
                        if *eigenvalue < 0.0 {
                            *eigenvalue = 0.0;
                        }
                    }
                    let projected_partial_phi_partial_f_square = eigendecom.recompose();
                    let small_hessian = &partial_f_partial_x
                        * projected_partial_phi_partial_f_square
                        * partial_f_partial_x.transpose();

                    return small_hessian;

            },
            HessianModification::InternalFlip => todo!(),
        }
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
        assert_eq!(x.len(), self.n_verts * 2);
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
        assert_eq!(x.len(), self.n_verts * 2);
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

    pub fn elastic_hessian<E: Energy<6, 2>>(
        &self,
        x: &DVector<f64>,
        energy: &E,
        modification: HessianModification,
    ) -> DMatrix<f64> {
        assert_eq!(x.len(), self.n_verts * 2);
        let mut res = DMatrix::zeros(x.len(), x.len());
        for i in 0..self.n_prims {
            let indices = self.get_indices(i);
            let mut vert_vec = SVector::<f64, 6>::zeros();
            vert_vec
                .iter_mut()
                .zip(indices.iter())
                .for_each(|(g_i, i)| *g_i = x[*i]);
            let small_hessian;
            match modification {
                HessianModification::NoModification => {
                    let energy: Hessian<6> = self.prim_energy(i, energy, vert_vec);
                    small_hessian = energy.hessian();
                }
                HessianModification::RemoveMinusEigenvalues => {
                    let energy: Hessian<6> = self.prim_energy(i, energy, vert_vec);
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
                    let energy: Hessian<6> = self.prim_energy(i, energy, vert_vec);
                    let hessian = energy.hessian();
                    let mut eigendecom = hessian.symmetric_eigen();
                    for eigenvalue in eigendecom.eigenvalues.iter_mut() {
                        if *eigenvalue < 0.0 {
                            *eigenvalue *= -1.0;
                        }
                    }
                    small_hessian = eigendecom.recompose();
                }
                HessianModification::InternalRemove => {
                    let vert_gradient_vec = Hessian::<6>::as_myscalar_vec(vert_vec);
                    let inv_mat = self.ma_invs[i];
                    let inv_mat = Hessian::<6>::as_constant_mat(inv_mat);
                    let square = self.volumes[i];
                    let mat = nalgebra::matrix![
                        vert_gradient_vec[4]-vert_gradient_vec[0], vert_gradient_vec[2]-vert_gradient_vec[0];
                        vert_gradient_vec[5]-vert_gradient_vec[1], vert_gradient_vec[3]-vert_gradient_vec[1];
                    ];
                    let matrix_f = mat * inv_mat;
                    println!("{matrix_f}");
                    let mut partial_f_partial_x = SMatrix::<f64, 6, 4>::zeros();

                    partial_f_partial_x
                        .column_mut(0)
                        .copy_from(&matrix_f[(0, 0)].gradient);
                    partial_f_partial_x
                        .column_mut(1)
                        .copy_from(&matrix_f[(1, 0)].gradient);

                    partial_f_partial_x
                        .column_mut(2)
                        .copy_from(&matrix_f[(0, 1)].gradient);
                    partial_f_partial_x
                        .column_mut(3)
                        .copy_from(&matrix_f[(1, 1)].gradient);

                    let mut matrix_f_clone = nalgebra::matrix![
                        Hessian::<4>::zero(),Hessian::<4>::zero();
                        Hessian::<4>::zero(),Hessian::<4>::zero();
                    ];

                    matrix_f_clone[(0, 0)].value = matrix_f[(0, 0)].value;
                    matrix_f_clone[(1, 0)].value = matrix_f[(1, 0)].value;
                    matrix_f_clone[(0, 1)].value = matrix_f[(0, 1)].value;
                    matrix_f_clone[(1, 1)].value = matrix_f[(1, 1)].value;

                    matrix_f_clone[(0, 0)].gradient[0] = 1.0;
                    matrix_f_clone[(1, 0)].gradient[1] = 1.0;
                    matrix_f_clone[(0, 1)].gradient[2] = 1.0;
                    matrix_f_clone[(1, 1)].gradient[3] = 1.0;

                    let i1 = matrix_f_clone.transpose() * matrix_f_clone;
                    let i1 = i1.trace();
                    // let i2=  ?`
                    // Is i2 used in formula?

                    // i3 = matrix_f.determinate()
                    let i3 = matrix_f_clone[(0, 0)] * matrix_f_clone[(1, 1)]
                        - matrix_f_clone[(0, 1)] * matrix_f_clone[(1, 0)];

                    // invertion test
                    // assert!(i3 > 0.0)
                    let i3 = i3 * i3;

                    let logi3 = autodiff::MyLog::myln(i3);

                    let ene = (i1 - logi3 - 2.0) * (energy.mu() / 2.0)
                        + logi3 * logi3 * (energy.lambda() / 8.0);
                    let ene = ene * (<Hessian<4> as num::One>::one() * square);
                    let partial_phi_partial_f_square = ene.hessian();

                    let mut eigendecom = partial_phi_partial_f_square.symmetric_eigen();
                    for eigenvalue in eigendecom.eigenvalues.iter_mut() {
                        if *eigenvalue < 0.0 {
                            *eigenvalue = 0.0;
                        }
                    }
                    let projected_partial_phi_partial_f_square = eigendecom.recompose();
                    small_hessian = &partial_f_partial_x
                        * projected_partial_phi_partial_f_square
                        * partial_f_partial_x.transpose();
                }
                HessianModification::InternalFlip => {
                    let vert_gradient_vec = Hessian::<6>::as_myscalar_vec(vert_vec);
                    let inv_mat = self.ma_invs[i];
                    let inv_mat = Hessian::<6>::as_constant_mat(inv_mat);
                    let square = self.volumes[i];
                    let mat = nalgebra::matrix![
                        vert_gradient_vec[4]-vert_gradient_vec[0], vert_gradient_vec[2]-vert_gradient_vec[0];
                        vert_gradient_vec[5]-vert_gradient_vec[1], vert_gradient_vec[3]-vert_gradient_vec[1];
                    ];
                    let matrix_f = mat * inv_mat;
                    println!("{matrix_f}");
                    let mut partial_f_partial_x = SMatrix::<f64, 6, 4>::zeros();

                    partial_f_partial_x
                        .column_mut(0)
                        .copy_from(&matrix_f[(0, 0)].gradient);
                    partial_f_partial_x
                        .column_mut(1)
                        .copy_from(&matrix_f[(1, 0)].gradient);

                    partial_f_partial_x
                        .column_mut(2)
                        .copy_from(&matrix_f[(0, 1)].gradient);
                    partial_f_partial_x
                        .column_mut(3)
                        .copy_from(&matrix_f[(1, 1)].gradient);

                    let mut matrix_f_clone = nalgebra::matrix![
                        Hessian::<4>::zero(),Hessian::<4>::zero();
                        Hessian::<4>::zero(),Hessian::<4>::zero();
                    ];

                    matrix_f_clone[(0, 0)].value = matrix_f[(0, 0)].value;
                    matrix_f_clone[(1, 0)].value = matrix_f[(1, 0)].value;
                    matrix_f_clone[(0, 1)].value = matrix_f[(0, 1)].value;
                    matrix_f_clone[(1, 1)].value = matrix_f[(1, 1)].value;

                    matrix_f_clone[(0, 0)].gradient[0] = 1.0;
                    matrix_f_clone[(1, 0)].gradient[1] = 1.0;
                    matrix_f_clone[(0, 1)].gradient[2] = 1.0;
                    matrix_f_clone[(1, 1)].gradient[3] = 1.0;

                    let i1 = matrix_f_clone.transpose() * matrix_f_clone;
                    let i1 = i1.trace();
                    // let i2=  ?`
                    // Is i2 used in formula?

                    // i3 = matrix_f.determinate()
                    let i3 = matrix_f_clone[(0, 0)] * matrix_f_clone[(1, 1)]
                        - matrix_f_clone[(0, 1)] * matrix_f_clone[(1, 0)];

                    // invertion test
                    // assert!(i3 > 0.0)
                    let i3 = i3 * i3;

                    let logi3 = autodiff::MyLog::myln(i3);

                    let ene = (i1 - logi3 - 2.0) * (energy.mu() / 2.0)
                        + logi3 * logi3 * (energy.lambda() / 8.0);
                    let ene = ene * (<Hessian<4> as num::One>::one() * square);
                    let partial_phi_partial_f_square = ene.hessian();

                    let mut eigendecom = partial_phi_partial_f_square.symmetric_eigen();
                    for eigenvalue in eigendecom.eigenvalues.iter_mut() {
                        if *eigenvalue < 0.0 {
                            *eigenvalue *= -1.0;
                        }
                    }
                    let projected_partial_phi_partial_f_square = eigendecom.recompose();
                    small_hessian = &partial_f_partial_x
                        * projected_partial_phi_partial_f_square
                        * partial_f_partial_x.transpose();
                }
            }
            for i in 0..6 {
                for j in 0..6 {
                    res[(indices[i], indices[j])] += small_hessian[(i, j)];
                }
            }
        }
        res
    }
}
