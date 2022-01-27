use nalgebra::{DVector, SMatrix};

use std::fs::File;
use std::io::Write;
use std::path::Path;
mod dim2;
mod dim3;
pub use dim2::*;
pub use dim3::*;
use std::collections::HashSet;

// D = T - 1, because rust doesn't support const generic operations yet
#[derive(Clone)]
pub struct Mesh<const D: usize, const T: usize> {
    // basic information
    pub n_verts: usize,
    pub n_prims: usize,

    // optional, only valid for 3d mesh
    // for 2d mesh, can be directly accessed by prim_connected_vert_indices
    pub surface: Option<HashSet<[usize; D]>>,

    // attributes for each vertex coordiants
    // length = n_verts* self.dim()
    pub verts: DVector<f64>,
    pub velos: DVector<f64>,
    pub accls: DVector<f64>,
    pub masss: DVector<f64>,

    // attributes for each face, no need to use DVector here, Vec is enough
    // length = n_prims
    pub volumes: Vec<f64>,
    pub ma_invs: Vec<SMatrix<f64, D, D>>,

    // connectivity information
    pub prim_connected_vert_indices: Vec<[usize; T]>,
    pub vert_connected_prim_indices: Vec<Vec<usize>>,
}

pub type Mesh2d = Mesh<2, 3>;
pub type Mesh3d = Mesh<3, 4>;

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
}

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
}

/// calculates the area of triangle
#[inline]
pub fn area(x1: f64, y1: f64, x2: f64, y2: f64, x3: f64, y3: f64) -> f64 {
    0.5 * ((x1 - x3) * (y2 - y1) - (x1 - x2) * (y3 - y1)).abs()
}

/// calcuates the volume of tet
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn volume(
    x1: f64,
    y1: f64,
    z1: f64,
    x2: f64,
    y2: f64,
    z2: f64,
    x3: f64,
    y3: f64,
    z3: f64,
    x4: f64,
    y4: f64,
    z4: f64,
) -> f64 {
    ((x4 - x1) * ((y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1))
        + (y4 - y1) * ((z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1))
        + (z4 - z1) * ((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)))
        / 6.0
}
