use std::mem::MaybeUninit;

use nalgebra::{DVector, Dim, OMatrix, RawStorage, RawStorageMut};

use crate::*;
#[derive(Debug, Clone)]
pub struct Csc64 {
    // The c code use size_t, which is an alias of u_long on my computer,
    // rust convert it as u64,
    // not std::os::raw::c_size_t, which is usize actually
    pub ncols: u64,
    pub nrows: u64,
    pub nnzs: u64,
    pub col_offsets: Vec<std::os::raw::c_int>,
    pub row_indices: Vec<std::os::raw::c_int>,
    pub values: Vec<std::os::raw::c_double>,
}

impl<R, C, S> From<&nalgebra::Matrix<f64, R, C, S>> for Csc64
where
    R: Dim,
    C: Dim,
    S: RawStorage<f64, R, C>,
{
    fn from(dense: &nalgebra::Matrix<f64, R, C, S>) -> Self {
        let mut col_offsets = Vec::with_capacity(dense.ncols() + 1);
        let mut row_idx = Vec::new();
        let mut values = Vec::new();

        col_offsets.push(0);
        for j in 0..dense.ncols() {
            for i in j..dense.nrows() {
                unsafe {
                    let v = dense.get_unchecked((i, j));
                    if *v != 0.0 {
                        row_idx.push(i as i32);
                        values.push(v.clone());
                    }
                }
            }
            col_offsets.push(row_idx.len() as i32);
        }

        Csc64 {
            ncols: dense.ncols() as u64,
            nrows: dense.nrows() as u64,
            nnzs: values.len() as u64,
            col_offsets,
            row_indices: row_idx,
            values,
        }
    }
}

pub unsafe fn csc_convert(a: &mut Csc64, c: *mut cholmod_common) -> cholmod_sparse {
    let b: MaybeUninit<cholmod_sparse> = MaybeUninit::uninit();
    let mut b = b.assume_init();
    b.nrow = a.nrows;
    b.ncol = a.ncols;
    b.nzmax = a.nnzs;

    b.p = a.col_offsets.as_ptr() as *mut _;
    b.i = a.row_indices.as_ptr() as *mut _;
    b.x = a.values.as_ptr() as *mut _;
    b.nz = std::ptr::null_mut();
    b.z = std::ptr::null_mut();

    // 0: un-sym
    // <0: lower triangle
    b.stype = -1;

    // 1: real
    b.xtype = 1;

    //0: int,  2: double
    b.itype = 0;

    // 0 is double, 1 is float
    b.dtype = 0;
    b.packed = 1;
    b.sorted = 1;

    let b_ptr: *mut cholmod_sparse = &mut b;
    assert_eq!(cholmod_check_sparse(b_ptr, c), 1);
    b
}

pub unsafe fn dense_convert<R: Dim, C: Dim, S: RawStorageMut<f64, R, C>>(
    a: &mut nalgebra::Matrix<f64, R, C, S>,
    c: *mut cholmod_common,
) -> cholmod_dense {
    let b: MaybeUninit<cholmod_dense> = MaybeUninit::uninit();
    let mut b = b.assume_init();

    b.ncol = a.ncols() as u64;
    b.nrow = a.nrows() as u64;

    b.nzmax = (a.ncols() * a.nrows()) as u64;

    // leading dimension
    b.d = a.nrows() as u64;
    b.x = a.data.ptr_mut() as *mut _;

    b.z = std::ptr::null_mut();

    b.xtype = 1;
    b.dtype = 0;

    let b_ptr: *mut cholmod_dense = &mut b;
    assert_eq!(cholmod_check_dense(b_ptr, c), 1);

    b
}

pub unsafe fn cholmod_dense_convert(a: *mut cholmod_dense, c: *mut cholmod_common) -> DVector<f64> {
    assert_eq!((*a).ncol, 1);
    assert_eq!(cholmod_check_dense(a, c), 1);
    let mut res = DVector::<f64>::zeros((*a).nrow as usize);
    let data_ptr = (*a).x as *mut f64;
    for i in 0..(*a).nrow {
        res[i as usize] = *data_ptr.add(i as usize);
    }
    res
}
