use crate::volume;
use crate::Mesh3d;
use nalgebra::matrix;
use nalgebra::DVector;
use nalgebra::SMatrix;
use rand::Rng;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use std::usize;

fn volume_mass_construct(
    density: f64,
    verts: &DVector<f64>,
    prim_connected_vert_indices: &[[usize; 4]],
) -> (DVector<f64>, Vec<f64>, Vec<SMatrix<f64, 3, 3>>) {
    let n_prims = prim_connected_vert_indices.len();

    let mut mass = DVector::<f64>::zeros(verts.len());
    let mut volumes = Vec::<f64>::new();
    let mut ma_invs = Vec::<SMatrix<f64, 3, 3>>::new();

    volumes.reserve_exact(n_prims);
    ma_invs.reserve_exact(n_prims);

    for [i, j, k, t] in prim_connected_vert_indices.iter() {
        let size = volume(
            verts[*i * 3],
            verts[*i * 3 + 1],
            verts[*i * 3 + 2],
            verts[*j * 3],
            verts[*j * 3 + 1],
            verts[*j * 3 + 2],
            verts[*k * 3],
            verts[*k * 3 + 1],
            verts[*k * 3 + 2],
            verts[*t * 3],
            verts[*t * 3 + 1],
            verts[*t * 3 + 2],
        );
        mass[*i * 3] += 0.25 * size * density;
        mass[*i * 3 + 1] += 0.25 * size * density;
        mass[*i * 3 + 2] += 0.25 * size * density;

        mass[*j * 3] += 0.25 * size * density;
        mass[*j * 3 + 1] += 0.25 * size * density;
        mass[*j * 3 + 2] += 0.25 * size * density;

        mass[*k * 3] += 0.25 * size * density;
        mass[*k * 3 + 1] += 0.25 * size * density;
        mass[*k * 3 + 2] += 0.25 * size * density;

        mass[*t * 3] += 0.25 * size * density;
        mass[*t * 3 + 1] += 0.25 * size * density;
        mass[*t * 3 + 2] += 0.25 * size * density;

        volumes.push(size);
        let matrix = matrix![
            verts[*j * 3 + 0] - verts[*i * 3 + 0], verts[*k * 3 + 0] - verts[*i * 3 + 0], verts[*t * 3 + 0] - verts[*i * 3 + 0] ;
            verts[*j * 3 + 1] - verts[*i * 3 + 1], verts[*k * 3 + 1] - verts[*i * 3 + 1], verts[*t * 3 + 1] - verts[*i * 3 + 1] ;
            verts[*j * 3 + 2] - verts[*i * 3 + 2], verts[*k * 3 + 2] - verts[*i * 3 + 2], verts[*t * 3 + 2] - verts[*i * 3 + 2] ;
        ];
        ma_invs.push(matrix.try_inverse().unwrap());
    }
    (mass, volumes, ma_invs)
}

// The output is wrapped in a Result to allow matching on errors
// Returns an Iterator to the Reader of the lines of the file.
fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

pub fn armadillo() -> Mesh3d {
    let ele_lines = read_lines("data/armadillo_4k.node").unwrap();
    let mut verts = DVector::<f64>::zeros(1180 * 3);
    let mut count = 0;
    for line in ele_lines {
        let line = line.unwrap();
        let mut words = line.split_whitespace();
        words.next();
        verts[count * 3] = words.next().unwrap().parse::<f64>().unwrap();
        verts[count * 3 + 1] = words.next().unwrap().parse::<f64>().unwrap();
        verts[count * 3 + 2] = words.next().unwrap().parse::<f64>().unwrap();
        count += 1;
    }
    assert_eq!(count, 1180);

    let ele_lines = read_lines("data/armadillo_4k.ele").unwrap();
    let mut prim_connected_vert_indices = Vec::<[usize; 4]>::new();
    let mut vert_connected_prim_indices = vec![Vec::<usize>::new(); 1180];
    let mut count = 0;

    for line in ele_lines {
        let line = line.unwrap();
        let mut words = line.split_whitespace();
        words.next();
        let v1 = words.next().unwrap().parse::<usize>().unwrap();
        let v2 = words.next().unwrap().parse::<usize>().unwrap();
        let v3 = words.next().unwrap().parse::<usize>().unwrap();
        let v4 = words.next().unwrap().parse::<usize>().unwrap();
        prim_connected_vert_indices.push([v1, v2, v3, v4]);

        vert_connected_prim_indices[v1].push(count);
        vert_connected_prim_indices[v2].push(count);
        vert_connected_prim_indices[v3].push(count);
        vert_connected_prim_indices[v4].push(count);
        count += 1
    }

    let density = 1e3;

    let (masss, volumes, ma_invs) =
        volume_mass_construct(density, &verts, &prim_connected_vert_indices);

    Mesh3d {
        n_verts: 1180,
        n_prims: 3717,
        verts,
        velos: DVector::<f64>::zeros(1180 * 3),
        accls: DVector::<f64>::zeros(1180 * 3),
        masss,
        volumes,
        ma_invs,

        prim_connected_vert_indices,
        vert_connected_prim_indices,
    }
}

pub fn cube(
    r: usize, // row
    c: usize, // column
    s: usize, // stack
    w: Option<f64>,
    h: Option<f64>,
    l: Option<f64>,
    d: Option<f64>, // disnety
    uniform: bool,
) -> Mesh3d {
    assert!(r * c > 0);
    //  the shape of this Plane

    // r-1 ----------r-1*c-1
    //     .............
    //     |   |   |   |
    // 1   -------------
    //     c  c+1...
    //     |   |   |   |
    // 0   -------------
    //     0   1   2   c-1

    // The structure of the cube is many layers of plane

    let get_index =
        |r_ind: usize, c_ind: usize, s_ind: usize| -> usize { s_ind * r * c + r_ind * r + c_ind };
    let w = w.unwrap_or(1.0);
    let h = h.unwrap_or(1.0);
    let l = l.unwrap_or(1.0);

    let mut rng = rand::thread_rng();
    let mut row_cooridantes: Vec<f64>;
    let mut col_cooridantes: Vec<f64>;
    let mut sta_cooridantes: Vec<f64>;
    if !uniform {
        row_cooridantes = (0..r - 1)
            .map(|_| ((r - 1) as f64) * w * rng.gen::<f64>())
            .collect();
        col_cooridantes = (0..c - 1)
            .map(|_| ((c - 1) as f64) * h * rng.gen::<f64>())
            .collect();
        sta_cooridantes = (0..s - 1)
            .map(|_| ((s - 1) as f64) * l * rng.gen::<f64>())
            .collect();
        row_cooridantes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        col_cooridantes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sta_cooridantes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        row_cooridantes[0] = 0.0;
        col_cooridantes[0] = 0.0;
        sta_cooridantes[0] = 0.0;
        row_cooridantes.push((r as f64 - 1.0) * w);
        col_cooridantes.push((c as f64 - 1.0) * h);
        sta_cooridantes.push((s as f64 - 1.0) * l)
    } else {
        row_cooridantes = (0..r).map(|i| (i as f64) * w).collect();
        col_cooridantes = (0..c).map(|i| (i as f64) * h).collect();
        sta_cooridantes = (0..s).map(|i| (i as f64) * l).collect();
    }
    let verts = DVector::from_fn(3 * r * c * s, |i, _| {
        if i % 3 == 0 {
            // row
            row_cooridantes[(i / 3) % (r * c) % r]
        } else if i % 3 == 1 {
            // column
            col_cooridantes[((i - 1) / 3) % (r * c) / r]
        } else {
            // stack
            sta_cooridantes[(((i - 2) / 3) / (r * c))]
        }
    });

    let mut prim_connected_vert_indices = Vec::<[usize; 4]>::new();
    let mut vert_connected_prim_indices = vec![Vec::<usize>::new(); r * c * s];
    prim_connected_vert_indices.reserve_exact(3 * (r - 1) * (c - 1) * (s - 1));
    let mut count = 0;

    // structure from https://www.researchgate.net/profile/Robert-Ban/publication/353522080/figure/fig1/AS:1050541652189184@1627480056506/Cube-decomposed-into-six-congruent-tetrahedra.ppm
    for i in 0..c - 1 {
        for j in 0..r - 1 {
            for k in 0..s - 1 {
                let a100 = get_index(i, j, k);
                let a110 = get_index(i + 1, j, k);
                let a000 = get_index(i, j + 1, k);
                let a101 = get_index(i, j, k + 1);
                let a111 = get_index(i + 1, j, k + 1);
                let a011 = get_index(i + 1, j + 1, k + 1);
                let a001 = get_index(i, j + 1, k + 1);
                let a010 = get_index(i + 1, j + 1, k);

                // orange
                prim_connected_vert_indices.push([a101, a100, a111, a000]);

                vert_connected_prim_indices[a101].push(count);
                vert_connected_prim_indices[a100].push(count);
                vert_connected_prim_indices[a111].push(count);
                vert_connected_prim_indices[a000].push(count);
                count += 1;

                // purple

                prim_connected_vert_indices.push([a000, a110, a010, a111]);

                vert_connected_prim_indices[a000].push(count);
                vert_connected_prim_indices[a110].push(count);
                vert_connected_prim_indices[a010].push(count);
                vert_connected_prim_indices[a111].push(count);
                count += 1;

                // blue

                prim_connected_vert_indices.push([a000, a111, a011, a010]);

                vert_connected_prim_indices[a000].push(count);
                vert_connected_prim_indices[a111].push(count);
                vert_connected_prim_indices[a011].push(count);
                vert_connected_prim_indices[a010].push(count);
                count += 1;
                // red

                prim_connected_vert_indices.push([a100, a110, a111, a000]);

                vert_connected_prim_indices[a100].push(count);
                vert_connected_prim_indices[a110].push(count);
                vert_connected_prim_indices[a111].push(count);
                vert_connected_prim_indices[a000].push(count);
                count += 1;
                // yellow

                prim_connected_vert_indices.push([a101, a000, a111, a001]);

                vert_connected_prim_indices[a101].push(count);
                vert_connected_prim_indices[a000].push(count);
                vert_connected_prim_indices[a111].push(count);
                vert_connected_prim_indices[a001].push(count);
                count += 1;

                // green

                prim_connected_vert_indices.push([a000, a111, a011, a001]);

                vert_connected_prim_indices[a000].push(count);
                vert_connected_prim_indices[a111].push(count);
                vert_connected_prim_indices[a011].push(count);
                vert_connected_prim_indices[a001].push(count);
                count += 1;
            }
        }
    }

    let density = d.unwrap_or(1e3);
    let (masss, volumes, ma_invs) =
        volume_mass_construct(density, &verts, &prim_connected_vert_indices);

    Mesh3d {
        n_verts: verts.len() / 3,
        n_prims: prim_connected_vert_indices.len(),

        velos: DVector::<f64>::zeros(verts.len()),
        accls: DVector::<f64>::zeros(verts.len()),
        verts,
        masss,
        volumes,
        ma_invs,

        prim_connected_vert_indices,
        vert_connected_prim_indices,
    }
}
