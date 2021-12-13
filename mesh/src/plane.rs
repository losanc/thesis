use crate::MeshType;
// use autodiff as ad;
use na::{DVector, SMatrix, SVector};
use nalgebra as na;
use nalgebra_sparse as nas;
use nas::CooMatrix;
use num::{One, Zero};
use std::fs::File;
use std::io::Write;
use std::path::Path;

const DENSITY: f64 = 1e3;

pub struct Plane {
    n_fixed: usize,
    n_verts: usize,
    n_trias: usize,
    vers: Vec<SVector<f64, 2>>,
    vels: Vec<SVector<f64, 2>>,
    accs: Vec<SVector<f64, 2>>,
    indices: Vec<[usize; 3]>,
    mass: Vec<f64>,
    square: Vec<f64>,
    m_inv: Vec<SMatrix<f64, 2, 2>>,
}

// macro_rules! energy_function {
//     ($vec:ident, $ene:ident,$mat:ident,$inv_mat:ident, $square:expr, $type:ty) => {
//         let $mat = na::matrix![
//                 $vec[4]-$vec[0], $vec[2]-$vec[0];
//                 $vec[5]-$vec[1], $vec[3]-$vec[1];
//             ];
//         let $mat = $mat*$inv_mat;
//         let $mat  = ($mat.transpose() * $mat -
//         na::matrix![
//             <$type>::one(), <$type>::zero();
//             <$type>::zero(), <$type>::one();
//         ])
//         *(<$type>::one()*0.5);

//         let $ene = ($mat.transpose()*$mat).trace()*(<$type>::one()*MIU) +
//          $mat.trace()*$mat.trace()*(<$type>::one()*(0.5*LAMBDA));
//          let $ene = $ene *(<$type>::one()*$square);

//     };
// }

impl MeshType<2, 3> for Plane {
    // type PriType = f64;
    // type PriVecType = SVector<f64, 6>;
    // type GradientPriType = ad::Gradient<6>;
    // type GradientPriVecType = SVector<ad::Gradient<6>, 6>;
    // type HessianPriType = ad::Hessian<6>;
    // type HessianPriVecType = SVector<ad::Hessian<6>, 6>;
    type MassMatrixType = CooMatrix<f64>;
    #[inline]
    fn n_verts(&self) -> usize {
        self.n_verts
    }
    #[inline]
    fn n_fixed_verts(&self) -> usize {
        self.n_fixed
    }
    #[inline]
    fn n_pris(&self) -> usize {
        self.n_trias
    }
    #[inline]
    fn m_inv(&self, i: usize) -> SMatrix<f64, 2, 2> {
        self.m_inv[i].clone()
    }
    #[inline]
    fn indices(&self) -> Vec<[usize; 3]> {
        self.indices.clone()
    }
    fn save_to_obj<P: AsRef<Path>>(&self, path: P) {
        let mut file = File::create(path).unwrap();
        writeln!(file, "g obj").unwrap();
        for vert in self.vers.iter() {
            writeln!(file, "v  {}  {}  {} ", vert[0], vert[1], 0.0).unwrap();
        }
        writeln!(file).unwrap();
        for inds in self.indices.iter() {
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

    #[inline]
    fn primitive_to_ind_vector(&self, index: usize) -> Vec<usize> {
        let vertex_index_3 = self.indices[index];
        vec![
            vertex_index_3[0] * 2,
            vertex_index_3[0] * 2 + 1,
            vertex_index_3[1] * 2,
            vertex_index_3[1] * 2 + 1,
            vertex_index_3[2] * 2,
            vertex_index_3[2] * 2 + 1,
        ]
    }

    fn all_vertices_to_vector(&self) -> DVector<f64> {
        let mut res = DVector::<f64>::zeros(self.n_verts * 2);
        for (i, v) in self.vers.iter().enumerate() {
            res[i * 2] = v[0];
            res[i * 2 + 1] = v[1];
        }
        res
    }

    fn all_velocities_to_vector(&self) -> DVector<f64> {
        let mut res = DVector::<f64>::zeros(self.n_verts * 2);
        for (i, v) in self.vels.iter().enumerate() {
            res[i * 2] = v[0];
            res[i * 2 + 1] = v[1];
        }
        res
    }

    fn set_all_vertices_vector(&mut self, vec: na::DVector<f64>) {
        for (i, v) in self.vers.iter_mut().enumerate() {
            *v = na::vector![vec[i * 2], vec[i * 2 + 1]];
        }
    }

    fn set_all_velocities_vector(&mut self, vec: na::DVector<f64>) {
        for (i, v) in self.vels.iter_mut().enumerate() {
            *v = na::vector![vec[i * 2], vec[i * 2 + 1]];
        }
    }

    fn mass_matrix(&self) -> Self::MassMatrixType {
        let row_indices = (0..self.n_verts * 2).collect::<Vec<usize>>();
        let col_indices = (0..self.n_verts * 2).collect::<Vec<usize>>();
        let values = self
            .mass
            .iter()
            .map(|x| vec![*x, *x])
            .flatten()
            .collect::<Vec<f64>>();
        CooMatrix::try_from_triplets(
            self.n_verts * 2,
            self.n_verts * 2,
            row_indices,
            col_indices,
            values,
        )
        .unwrap()
    }

    // fn primitive_elastic_energy(&self, index: usize) -> Self::PriType {
    //     let vector = self.primitive_to_vert_vector(index);
    //     let inv_mat = self.m_inv[index];
    //     energy_function!(vector, ene, mat, inv_mat, self.square[index], Self::PriType);
    //     ene
    // }

    // fn primitive_elastic_energy_gradient(&self, index: usize) -> Self::GradientPriType {
    //     let vector = self.primitive_to_vert_vector(index);
    //     let mut vector: Self::GradientPriVecType = ad::vector_to_gradients(&vector);
    //     for i in 0..3 {
    //         if self.indices[index][i] < self.n_fixed {
    //             vector[2 * i].as_constant();
    //             vector[2 * i + 1].as_constant();
    //         }
    //     }
    //     let inv_mat = self.m_inv[index];
    //     let inv_mat = ad::constant_matrix_to_gradients(&inv_mat);

    //     energy_function!(
    //         vector,
    //         ene,
    //         mat,
    //         inv_mat,
    //         self.square[index],
    //         Self::GradientPriType
    //     );
    //     ene
    // }

    // fn primitive_elastic_energy_hessian(&self, index: usize) -> Self::HessianPriType {
    //     let vector = self.primitive_to_vert_vector(index);
    //     let mut vector: Self::HessianPriVecType = ad::vector_to_hessians(&vector);
    //     for i in 0..3 {
    //         if self.indices[index][i] < self.n_fixed {
    //             vector[2 * i].as_constant();
    //             vector[2 * i + 1].as_constant();
    //         }
    //     }

    //     let inv_mat = self.m_inv[index];
    //     let inv_mat = ad::constant_matrix_to_hessians(&inv_mat);

    //     energy_function!(
    //         vector,
    //         ene,
    //         mat,
    //         inv_mat,
    //         self.square[index],
    //         Self::HessianPriType
    //     );
    //     ene
    // }
}

#[inline]
// TODO
fn square_area(_coords: &[SVector<f64, 2>; 3]) -> f64 {
    0.5
}

impl Plane {
    pub fn new(r: usize, c: usize) -> Plane {
        //  the shape of this Plane
        // r -------------
        //   ..........r*c-1
        //   |   |   |   |
        // 1 -------------
        //   c  c+1...
        //   |   |   |   |
        // 0 -------------
        //   0   1   2   c-1

        let get_index = |r_in: usize, c_in: usize| -> usize { r_in * c + c_in };

        let mut vers = Vec::<SVector<f64, 2>>::new();

        for i in 0..r {
            for j in 0..c {
                vers.push(na::vector![i as f64, j as f64]);
            }
        }

        let mut indices = Vec::<[usize; 3]>::new();
        for i in 0..r - 1 {
            for j in 0..c - 1 {
                indices.push([get_index(i, j), get_index(i, j + 1), get_index(i + 1, j)]);
            }
        }

        for i in 1..r {
            for j in 1..c {
                indices.push([get_index(i, j), get_index(i, j - 1), get_index(i - 1, j)]);
            }
        }
        let mut mass = vec![0.0; r * c];
        let mut square = Vec::<f64>::new();
        let mut m_inv = Vec::<SMatrix<f64, 2, 2>>::new();
        square.reserve_exact(indices.len());
        m_inv.reserve_exact(indices.len());
        for [i, j, k] in indices.iter() {
            let size = square_area(&[vers[*i], vers[*j], vers[*k]]);
            mass[*i] += 0.333 * size * DENSITY;
            mass[*j] += 0.333 * size * DENSITY;
            mass[*k] += 0.333 * size * DENSITY;
            square.push(size);
            let matrix =
                SMatrix::<f64, 2, 2>::from_columns(&[vers[*k] - vers[*i], vers[*j] - vers[*i]]);
            m_inv.push(matrix.try_inverse().unwrap());
        }

        Plane {
            n_fixed: 1,
            n_verts: r * c,
            vers,
            vels: vec![na::vector![0.0, 0.0]; r * c],
            accs: vec![na::vector![0.0, 0.0]; r * c],
            n_trias: indices.len(),
            indices,
            mass,
            square,
            m_inv,
        }
    }
}
