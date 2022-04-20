use std::{borrow::Borrow, ops::IndexMut};

const DIM: usize = 10;

fn sparse_chol(cscMatrix: &nalgebra_sparse::CscMatrix<f64>, vec: &nalgebra::DVector<f64>) {
    let mut scchol = nalgebra_sparse::factorization::CscCholesky::factor(&cscMatrix).unwrap();

    let l = scchol.l_mut();

    let (col_offsets, row_indices, mut values) = l.clone().disassemble();

    let mut x_vec = vec.clone();

    println!("{}", x_vec);
    for k in 0..DIM {
        println!("{}", x_vec);
        let lkk = values.index_mut(col_offsets[k]);
        let xk = x_vec[k];
        if xk.abs() < 1e-5 {
            // println!("{k}");
            continue;
        }
        let lkkv = *lkk;
        let r = (lkkv * lkkv + xk * xk).sqrt();
        let c = r / lkkv;
        let s = xk / lkkv;
        *lkk = r;
        if k < DIM - 1 {
            let xslice = x_vec.index_mut((k + 1..DIM, 0));

            println!("xslice before {xslice}");
        }
        for m in col_offsets[k] + 1..col_offsets[k + 1] {
            let r_index = row_indices[m];
            let v = values.index_mut(m);
            let x_r_index = x_vec.index_mut(r_index);

            *v = (*v + s * *x_r_index) / c;

            *x_r_index *= c;
            *x_r_index -= s * *v;
        }
        if k < DIM - 1 {
            let xslice = x_vec.index_mut((k + 1..DIM, 0));

            println!("xslice before {xslice}");
        }
    }

    println!(
        "sparse {}",
        nalgebra::DMatrix::from(
            &(nalgebra_sparse::CscMatrix::try_from_csc_data(
                10,
                10,
                col_offsets.clone(),
                row_indices.clone(),
                values.clone()
            ))
            .unwrap()
        )
    );
}

fn main() {
    let vec = nalgebra::DVector::<f64>::new_random(DIM).abs() * 1e2;
    let mut matrix = nalgebra::DMatrix::from_diagonal(&vec);

    matrix[(3, 4)] = 0.0001;
    matrix[(4, 3)] = 0.0001;
    matrix[(3, 9)] = 0.0001;
    matrix[(9, 3)] = 0.0001;
    matrix[(4, 9)] = 0.0001;
    matrix[(9, 4)] = 0.0001;
    let cscmatrix = nalgebra_sparse::CscMatrix::from(&matrix);

    let mut vec = nalgebra::DVector::<f64>::new_random(DIM) * 1e2;

    vec.index_mut((0..3, 0)).fill(0.0);

    vec.index_mut((5..9, 0)).fill(0.0);

    sparse_chol(&cscmatrix, &vec);

    let sum = &matrix + &vec * vec.transpose();

    // println!("sum::  {sum}");

    let chol = matrix.cholesky().unwrap();

    let newchol = sum.clone().cholesky().unwrap();

    let mut old_l = chol.l();
    println!("first old l {:.5}", old_l);

    let mut vec_copy = vec.clone();

    for i in 0..DIM {
        // println!("real: {old_l}");
        println!("x {vec_copy}");

        let r = (old_l[(i, i)] * old_l[(i, i)] + vec_copy[i] * vec_copy[i]).sqrt();
        let c = r / old_l[(i, i)];
        let s = vec_copy[i] / old_l[(i, i)];
        old_l[(i, i)] = r;

        if i < DIM - 1 {
            let mut lslice = old_l.index_mut((i + 1..DIM, i));
            lslice += s * vec_copy.index((i + 1..DIM, 0));

            lslice /= c;

            let mut xslice = vec_copy.index_mut((i + 1..DIM, 0));
            println!("xslice before {xslice}");
            xslice *= c;
            xslice -= s * lslice;
            println!("xslice after {xslice}");
        }
    }
    println!("{:.5}", old_l);
}
