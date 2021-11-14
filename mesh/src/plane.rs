use crate::MeshType;
use autodiff as ad;
use na::{DVector, SMatrix, SVector};
use nalgebra as na;
use nalgebra_sparse as nas;
use nas::CscMatrix;
use num::{One, Zero};
use std::fs::File;
use std::io::Write;
use std::path::Path;

const DENSITY: f64 = 1e3;
const E: f64 = 1e6;
const NU: f64 = 0.33;
const MIU: f64 = E / (2.0 * (1.0 + NU));
const LAMBDA: f64 = (E * NU) / ((1.0 + NU) * (1.0 - 2.0 * NU));

pub struct Plane {
    pub n_fixed: usize,
    pub n_verts: usize,
    pub n_trias: usize,
    pub vers: Vec<SVector<f64, 2>>,
    pub vels: Vec<SVector<f64, 2>>,
    pub accs: Vec<SVector<f64, 2>>,
    pub indices: Vec<[usize; 3]>,
    mass: Vec<f64>,
    square: Vec<f64>,
    m_inv: Vec<SMatrix<f64, 2, 2>>,
}

macro_rules! energy_function {
    ($vec:ident, $ene:ident,$mat:ident,$inv_mat:ident, $square:expr, $type:ty) => {
        let $mat = na::matrix![
                $vec[4]-$vec[0], $vec[2]-$vec[0];
                $vec[5]-$vec[1], $vec[3]-$vec[1];
            ];
        let $mat = $mat*$inv_mat;
        let $mat  = ($mat.transpose() * $mat -
        na::matrix![
            <$type>::one(), <$type>::zero();
            <$type>::zero(), <$type>::one();
        ])
        *(<$type>::one()*0.5);

        let $ene = ($mat.transpose()*$mat).trace()*(<$type>::one()*MIU) +
         $mat.trace()*$mat.trace()*(<$type>::one()*(0.5*LAMBDA));
         let $ene = $ene *(<$type>::one()*$square);

    };
}

impl MeshType for Plane {
    type PriType = f64;
    type PriVecType = SVector<f64, 6>;
    type GradientPriType = ad::Gradient<6>;
    type GradientPriVecType = SVector<ad::Gradient<6>, 6>;
    type HessianPriType = ad::Hessian<6>;
    type HessianPriVecType = SVector<ad::Hessian<6>, 6>;
    type MassMatrixType = CscMatrix<f64>;

    fn save_to_obj<P: AsRef<Path>>(&self, path: P) {
        let mut file = File::create(path).unwrap();
        write!(file, "g obj\n").unwrap();
        for vert in self.vers.iter() {
            write!(file, "v  {}  {}  {} \n", vert[0], vert[1], 0.0).unwrap();
        }
        write!(file, "\n").unwrap();
        for inds in self.indices.iter() {
            write!(
                file,
                "f  {}  {}  {} \n",
                inds[0] + 1,
                inds[1] + 1,
                inds[2] + 1
            )
            .unwrap();
        }
    }

    #[inline]
    fn primitive_to_vert_vector(&self, index: usize) -> Self::PriVecType {
        let vertex_index_3 = self.indices[index];
        na::vector![
            self.vers[vertex_index_3[0]][0],
            self.vers[vertex_index_3[0]][1],
            self.vers[vertex_index_3[1]][0],
            self.vers[vertex_index_3[1]][1],
            self.vers[vertex_index_3[2]][0],
            self.vers[vertex_index_3[2]][1]
        ]
    }

    fn all_to_vert_vector(&self) -> DVector<f64> {
        let mut res = DVector::<f64>::zeros(self.n_verts * 2);
        for (i, v) in self.vers.iter().enumerate() {
            res[i * 2] = v[0];
            res[i * 2 + 1] = v[1];
        }
        res
    }

    fn all_vels_to_vert_vector(&self) -> DVector<f64> {
        let mut res = DVector::<f64>::zeros(self.n_verts * 2);
        for (i, v) in self.vels.iter().enumerate() {
            res[i * 2] = v[0];
            res[i * 2 + 1] = v[1];
        }
        res
    }

    fn set_all_to_vert_vector(&mut self, vec: &na::DVector<f64>) {
        for (i, v) in self.vers.iter_mut().enumerate() {
            *v = na::vector![vec[i * 2], vec[i * 2 + 1]];
        }
    }

    fn set_all_to_velo_vector(&mut self, vec: &na::DVector<f64>) {
        for (i, v) in self.vels.iter_mut().enumerate() {
            *v = na::vector![vec[i * 2], vec[i * 2 + 1]];
        }
    }

    fn mass_matrix(&self) -> Self::MassMatrixType {
        let row_indices = (0..self.n_verts * 2).collect::<Vec<usize>>();
        let col_offsets = (0..self.n_verts * 2 + 1).collect::<Vec<usize>>();
        let values = self
            .mass
            .iter()
            .map(|x| vec![*x, *x])
            .flatten()
            .collect::<Vec<f64>>();
        CscMatrix::try_from_csc_data(
            self.n_verts * 2,
            self.n_verts * 2,
            col_offsets,
            row_indices,
            values,
        )
        .unwrap()
    }

    fn primitive_elastic_energy(&self, index: usize) -> Self::PriType {
        let vector = self.primitive_to_vert_vector(index);
        let inv_mat = self.m_inv[index];
        energy_function!(vector, ene, mat, inv_mat, self.square[index], Self::PriType);
        ene
    }

    fn primitive_elastic_energy_gradient(&self, index: usize) -> Self::GradientPriType {
        let vector = self.primitive_to_vert_vector(index);
        let mut vector: Self::GradientPriVecType = ad::vector_to_gradients(&vector);
        for i in 0..3 {
            if self.indices[index][i] < self.n_fixed {
                vector[2 * i].to_constant();
                vector[2 * i + 1].to_constant();
            }
        }
        let inv_mat = self.m_inv[index];
        let inv_mat = ad::constant_matrix_to_gradients(&inv_mat);

        energy_function!(
            vector,
            ene,
            mat,
            inv_mat,
            self.square[index],
            Self::GradientPriType
        );
        ene
    }

    fn primitive_elastic_energy_hessian(&self, index: usize) -> Self::HessianPriType {
        let vector = self.primitive_to_vert_vector(index);
        let mut vector: Self::HessianPriVecType = ad::vector_to_hessians(&vector);
        for i in 0..3 {
            if self.indices[index][i] < self.n_fixed {
                vector[2 * i].to_constant();
                vector[2 * i + 1].to_constant();
            }
        }

        let inv_mat = self.m_inv[index];
        let inv_mat = ad::constant_matrix_to_hessians(&inv_mat);

        energy_function!(
            vector,
            ene,
            mat,
            inv_mat,
            self.square[index],
            Self::HessianPriType
        );
        ene
    }
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
            n_fixed: c,
            n_verts: r * c,
            vers: vers,
            vels: vec![na::vector![0.0, 0.0]; r * c],
            accs: vec![na::vector![0.0, 0.0]; r * c],
            n_trias: indices.len(),
            indices: indices,
            mass: mass,
            square: square,
            m_inv: m_inv,
        }
    }
}
