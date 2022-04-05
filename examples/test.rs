use mesh::{plane, NeoHookean2d};
use nalgebra::matrix;

const E: f64 = 1e3;
const NU: f64 = 0.2;
const MIU: f64 = E / (2.0 * (1.0 + NU));
const LAMBDA: f64 = (E * NU) / ((1.0 + NU) * (1.0 - 2.0 * NU));
fn main() {
    let plane = plane(2, 2, None, None, None);

    let energy = NeoHookean2d {
        mu: MIU,
        lambda: LAMBDA,
    };
    let mut x = (&plane.verts).clone();
    println!("{}", x);
    let res = plane.elastic_hessian(&x, &energy);
    println!("{:.4}", res);

    let alpha: f64 = 10.0;
    let matrix = matrix![
        alpha.cos(),alpha.sin();
        -alpha.sin(), alpha.cos();
    ];
    for i in 0..plane.n_verts {
        let mut slice = x.index_mut((2 * i..2 * i + 2, 0));
        let res = &matrix * &slice;
        for j in 0..2 {
            slice[j] = res[j];
        }
    }
    println!("{}", x);
    let res = plane.elastic_hessian(&x, &energy);
    println!("{:.4}", res);
}
