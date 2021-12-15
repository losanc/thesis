pub use self::plane::Plane;
use nalgebra as na;
use std::path::Path;
mod plane;
use nalgebra::SMatrix;

// D = T+1
// because rust doesn't support const generic operation yet
pub trait MeshType<const T: usize, const D: usize> {
    // /// primitive elastic energy
    // type PriType;

    // /// primitive elastic energy wrt. to each vertex coordiante gradient
    // type GradientPriType;
    // /// primitive elastic energy wrt. to each vertex coordiante gradient
    // type HessianPriType;

    // /// vector of primitive vertex coordiante
    // type PriVecType;
    // /// vector of primitive vertex coordiante  as gradient
    // type GradientPriVecType;
    // /// vector of primitive vertex coordiante  as hessian
    // type HessianPriVecType;

    /// mass matrix type
    type MassMatrixType;

    #[inline]
    fn dim(&self) -> usize {
        T
    }

    fn indices(&self) -> Vec<[usize; D]>;
    fn n_verts(&self) -> usize;
    fn n_fixed_verts(&self) -> usize;
    fn n_pris(&self) -> usize;
    fn m_inv(&self, i: usize) -> SMatrix<f64, T, T>;
    fn volume(&self, i: usize) -> f64;

    /// get verteices vector of all vertices
    fn all_vertices_to_vector(&self) -> na::DVector<f64>;

    fn all_velocities_to_vector(&self) -> na::DVector<f64>;

    /// set verteices vector of all vertices
    fn set_all_vertices_vector(&mut self, vec: na::DVector<f64>);

    /// set velocities vector of all vertices
    fn set_all_velocities_vector(&mut self, vec: na::DVector<f64>);

    fn mass_matrix(&self) -> Self::MassMatrixType;

    /// get a primitive verteices vector of primitive
    fn primitive_to_ind_vector(&self, index: usize) -> Vec<usize>;

    /// save object to obj file
    fn save_to_obj<P: AsRef<Path>>(&self, path: P);

    // /// returns the energy of a primitive(e.g. tet or triangle) as f64 number
    // fn primitive_elastic_energy(&self, index: usize) -> Self::PriType;
    // /// returns the energy of a primitive(e.g. tet or triangle) as gradient
    // fn primitive_elastic_energy_gradient(&self, index: usize) -> Self::GradientPriType;
    // /// returns the energy of a primitive(e.g. tet or triangle) as hessian
    // fn primitive_elastic_energy_hessian(&self, index: usize) -> Self::HessianPriType;
}
