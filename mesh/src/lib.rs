use nalgebra::{DVector, SMatrix};
mod dim2;
mod dim3;
mod energy;
mod mesh_impl2d;
mod mesh_impl3d;
mod util;
pub use dim2::*;
pub use dim3::*;
pub use energy::*;
pub use util::*;

// D = T - 1, because rust doesn't support const generic operations yet
#[derive(Clone)]
pub struct Mesh<const D: usize, const T: usize> {
    // basic information
    pub n_verts: usize,
    pub n_prims: usize,

    // optional, only valid for 3d mesh
    // for 2d mesh, can be directly accessed by prim_connected_vert_indices
    pub surface: Option<Vec<[usize; D]>>,

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

pub type StVenantVirchhoff2d = StVenantVirchhoff<2>;
pub type StVenantVirchhoff3d = StVenantVirchhoff<3>;
pub type NeoHookean2d = NeoHookean<2>;
pub type NeoHookean3d = NeoHookean<3>;

#[derive(Debug, Clone, Copy)]
pub enum HessianModification {
    NoModification,
    RemoveMinusEigenvalues,
    FlipMinusEigenvalues,
}
