// use nalgebra::SMatrix;
// use nalgebra_sparse::CsrMatrix;

// pub mod my_newton;
// pub mod mymacro;
pub mod scenarios;
pub mod static_object;

// pub fn get_csr_index(matrix: &CsrMatrix<f64>, r: usize, c: usize) -> usize {
//     let (row_offsets, col_indices, _) = matrix.csr_data();
//     for i in row_offsets[r]..row_offsets[r + 1] {
//         if col_indices[i] == c {
//             return i;
//         }
//     }
//     panic!("shouldn't happen")
// }

// pub fn get_csr_index_matrix<const D: usize>(
//     matrix: &CsrMatrix<f64>,
//     indices: SMatrix<[usize; 2], D, D>,
// ) -> SMatrix<usize, D, D> {
//     let mut res = SMatrix::<usize, D, D>::zeros();
//     res.iter_mut()
//         .zip(indices.iter())
//         .for_each(|(r, i)| *r = get_csr_index(matrix, i[0], i[1]));
//     res
// }
