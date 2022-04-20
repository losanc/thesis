fn main() {
    let matrix = nalgebra::DMatrix::<f64>::new_random(5, 5).abs() * 1e2;
    let matrix = matrix.transpose() + matrix;

    let eigendecom = matrix.clone().symmetric_eigen();
    let mut res = nalgebra::DMatrix::<f64>::zeros(5, 5);
    for i in 0..5 {
        res += eigendecom.eigenvalues[i]
            * eigendecom.eigenvectors.column(i)
            * eigendecom.eigenvectors.column(i).transpose();
    }
    println!("{}", res - matrix);
}
