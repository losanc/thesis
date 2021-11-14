pub use self::plane::Plane;
use nalgebra as na;
use std::path::Path;
mod plane;

pub trait MeshType {
    /// primitive elastic energy
    type PriType;

    /// primitive elastic energy wrt. to each vertex coordiante gradient
    type GradientPriType;
    /// primitive elastic energy wrt. to each vertex coordiante gradient
    type HessianPriType;

    /// vector of primitive vertex coordiante
    type PriVecType;
    /// vector of primitive vertex coordiante  as gradient
    type GradientPriVecType;
    /// vector of primitive vertex coordiante  as hessian
    type HessianPriVecType;
    /// mass matrix type
    type MassMatrixType;

    /// save object to obj file
    fn save_to_obj<P: AsRef<Path>>(&self, path: P);

    /// get a primitive verteices vector of primitive
    fn primitive_to_vert_vector(&self, index: usize) -> Self::PriVecType;

    /// get verteices vector of all vertices
    fn all_to_vert_vector(&self) -> na::DVector<f64>;

    fn all_vels_to_vert_vector(&self) -> na::DVector<f64>;

    /// set verteices vector of all vertices
    fn set_all_to_vert_vector(&mut self, vec: &na::DVector<f64>);

    /// set velocities vector of all vertices
    fn set_all_to_velo_vector(&mut self, vec: &na::DVector<f64>);

    fn mass_matrix(&self) -> Self::MassMatrixType;

    /// returns the energy of a primitive(e.g. tet or triangle) as f64 number
    fn primitive_elastic_energy(&self, index: usize) -> Self::PriType;
    /// returns the energy of a primitive(e.g. tet or triangle) as gradient
    fn primitive_elastic_energy_gradient(&self, index: usize) -> Self::GradientPriType;
    /// returns the energy of a primitive(e.g. tet or triangle) as hessian
    fn primitive_elastic_energy_hessian(&self, index: usize) -> Self::HessianPriType;
}
