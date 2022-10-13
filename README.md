This is the source code of my master thesis `Fast Global Hessian Assembly in Finite Element Simulations`.

You can find the paper [here](./thesis.pdf).

## how to run it:

```bash
cargo run --release --example beam   --features log 1e3 0.33 0.01 1e3 20 80 1.0 0.125 0.0 0 name comment 200 no 10 true 20
```

this is the beam simulation in the thesis, and you can change the parameters as you want
```
1e3: Young's modulus
0.33: Poisson's ratio
0.01: time step
1e3: density
20: number of row
80: number of columns
1.0: damping ratio
0.125: size of the square
0.0: epsilon
0: k
name: name of the generated sequences
comment:  comment in the log file
200: frames
no: hessian modifications, options are no(no modification), flip(flip the negative eigenvalues) remove(truncate the negative eigenvalues),
10: precision of float number in the file name (doesn't matter very much)
true: uniform mesh or non-uniform
20: random seed
```

after that, you will see the log files in 'output/log', each folder is one simulation with the same physical parameters. For all the files inside one folder, are different epsilon and k

If you want to visualize the result, you can run it with  `cargo run --release --example beam   --features save` with the same other parameters. But `output/mesh` this folder must exist before running the code. After it, it will generate a sequence of mesh files in the folder `output/mesh`.

This is another circle simulation in the thesis

```bash
cargo run --example circle  --release --features log 1e4 0.33 0.01  1e3 1 25  1.0  0.0 0  name  nothing  20 no 10
```
 
```
1e4: Young's modulus
0.33: Poisson's ratio
0.01: time step
 
1e3: density
1: radius of the circle
25: resolution of the circle, see here for the algorithm
1.0: damping ratio
 
0.0: epsilon
0: k
name: name of the sequence
 
nothing: comment in the log file
20: number of frames
no: hessian modification
10: precision of float number in the file name (not very important)
```