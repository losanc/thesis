use matrixcompare::assert_matrix_eq;
use nalgebra::{DMatrix, DVector};
use nalgebra_sparse::CsrMatrix;
use optimization::*;
#[test]
fn test_ls() {
    // let l = 50;
    // let matrix = DMatrix::<f64>::new_random(l, l);
    // let csrmatrix = CsrMatrix::from(&matrix);
    // let v = DVector::<f64>::new_random(l);
    // let res1 = csrmatrix.mul(&v);
    // let res2 = matrix * v;
    // assert_matrix_eq!(res1, res2, comp = float);
}
