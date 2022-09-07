use mesh::{plane, NeoHookean2d};

pub fn main() {
    let mesh = plane(2, 2, None, None, None, true, 10);
    let e = 1e3;
    let nu = 0.3;
    let energy = NeoHookean2d {
        mu: e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)),
        lambda: e * nu / ((1.0 + nu) * 2.0),
    };
    let p = nalgebra::SVector::<f64,6>::new_random();
    let m1 = mesh.prim_projected_hessian(0,&energy,p, mesh::HessianModification::RemoveMinusEigenvalues);
    let m2 = mesh.prim_projected_hessian(0,&energy,p, mesh::HessianModification::InternalRemove);
    // let m3 = mesh.elastic_hessian(&p, &energy, mesh::HessianModification::NoModification);
    // println!("{:8}",m1-m2);
    println!("{}",m1.symmetric_eigenvalues());
    println!("{}",m2.symmetric_eigenvalues());
    // println!("{:.8}", &m1 - m2);
    // println!("{:.8}", &m1 - m3);
    // assert_matrix_eq!(m1,m2);
}
