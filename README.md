## Week 2 (12.16)


### Algorithm
It contains 2 parts, a total energy constructing 
```rust

fn global_gradien(x){
    let result;
    result += inertia.gradient();
    result += elastic.gradient();
    result += bounce.gradient();
    result
}

fn global_hessian(x){
    let result;
    result += inertia.hessian();
    result += elastic.hessian();
    result += bounce.hessian();
    result
}
```
and several small parts energy construcing.

Inertia and bouncing energy are pretty simple.
```rust
impl Inertia{
    fn gradient(x){
        mass_matrix * (x-x_tao-gravity) / (dt*dt) 
    }
    fn hessian(x){
        mass_matrix/ (dt*dt) 
    }
}

// coefficients for bouncing penalty energy
let keta = 1e8;
// define the energy as 
// if x>=0 return 0
// if x<0  return -keta*x*x*x
impl Bounce{
    fn gradient(x){
        let result;
        for (i,x_i) in x.enumerate().iter(){
            if x_i< 0
                result[i]-= keta * x_i*x_i;
        }
    }
    fn hessian(x){
        let result;
        for (i,x_i) in x.enumerate().iter(){
            if x_i< 0
                result[i,i]-= keta * x_i;
        }
    }
}
```
So the tricky part is about elastic energy
```rust
let record : Hashset<usize>;
struct Elastic{
    // a list of gradient vector for each triangle
    old_gradient: Vec<SVector<6>>
    // a list of hessian matrix for each triangle
    old_hessian: Vec<SMatrix<6,6>>
}

impl Elastic{
    fn gradient(x){
        for i in each_triangle{
            let this_triangle_gradient;
            if this_triangle_gradient
            .is_close_to(old_gradient(i)){
                record.insert(i);
            }
            update_old_gradient();
        }
    }
    fn hessian(x){
        for i in each_triangle{
            if record.contains(i){
                let this_triangle_hessian = old_hessian[i]
                // also it could compare the old_hessian[i] and real hessian
            }else{
                // calculate the real hessian
                update_old_hessian();
            }
        }
    }
}
```

### Idea:
Because the elastic energy is purly a function of vertex coordiantes `x`, so if the gradient is close to the old gradient (if necessary, we could make sure the energy is close to old energy as well), it could possibly mean that `x` is also close to the old coodiante(e.g. it's alreay in the energy minimum coodinate), in this case, hessian wouln't change very much either.


### problem:
1. Possible memory explosion
2. Only works with elastic energy (though other part is quite simple to evaluate the hessian)


## Week 1 (12.02)

### Algorithm
```rust
let record : Hashset<usize>;

fn global_gradien(x){
    // calculate global gradient
    let gradient = result;

    for (i,g) in gradient.enumerate().iter(){
        if g.abs()<1e-5{
            record.insert(i);
        }
    }
}

fn global_hessian(x){
    // calculate other part of hessian
    for i in each_triangle{
        if record.contains(all_3_vertices) {
            // skip this small hessian
            continue;
        }
    }

}
```

### Problem:
1. This is actually not the original idea
2. Impossible to work for cross time-step
3. This is equivalent to put a zero matrix in the global hessian, but the real hessian is not a a zero matrix. (Interestingly, this algorithm can work)