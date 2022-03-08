use autodiff::*;
use nalgebra::{matrix, vector};
#[test]
fn test_div1() {
    let a = vector![1.0, 2.0];
    let a = vector_to_hessians(a);
    let x = a[0];
    let y = a[1];
    let res = x * x + y * y;
    let res = res / x;
    // res = x+ y^2/x, where x=1, y=2
    assert_eq!(res.value(), 5.0);
    assert_eq!(res.gradient(), vector![-3.0, 4.0]);
    assert_eq!(res.hessian(), matrix![8.0,-4.0;-4.0,2.0]);
}

#[test]
fn test_div2() {
    let a = vector![1.0, 2.0];
    let a = vector_to_hessians(a);
    let x = a[0];
    let y = a[1];
    let res = x * x + y * y;
    let res = res / x;
    let res = res / 2.0;
    // res = (x+ y^2/x)/2, where x=1, y=2
    assert_eq!(res.value(), 2.5);
    assert_eq!(res.gradient(), vector![-1.5, 2.0]);
    assert_eq!(res.hessian(), matrix![4.0,-2.0;-2.0,1.0]);
}

#[test]
fn test_log1() {
    let a = vector![2.0, 3.0];
    let a = vector_to_gradients(a);
    let x = a[0];
    let y = a[1];
    let res = x * x + y * y;
    let res = res.myln();
    assert!((res.value() - 2.564949357461536).abs() < 1e-5);
    assert!((res.gradient() - vector![0.3077, 0.4615]).amax() < 1e-3);
}

#[test]
fn test_log2() {
    let a = vector![2.0, 3.0];
    let a = vector_to_hessians(a);
    let x = a[0];
    let y = a[1];
    let res = x * x + y * y;
    let res = res.myln();
    assert!((res.value() - 2.564949357461536).abs() < 1e-5);
    assert!((res.gradient() - vector![0.3077, 0.4615]).amax() < 1e-3);
    assert!(
        (res.hessian()
            - matrix![0.0592, -0.1420;
                    -0.1420,-0.0592;])
        .amax()
            < 1e-3
    );
}
