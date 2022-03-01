use crate::volume;
use crate::Mesh3d;
use nalgebra::matrix;
use nalgebra::DVector;
use nalgebra::SMatrix;
use std::collections::HashSet;
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
    let mut surface = HashSet::<[usize; 3]>::new();

    let sort = |(x, y, z)| -> [usize; 3] {
        if x < y && y < z {
            return [x, y, z];
        };
        if x < z && z < y {
            return [x, z, y];
        };
        if y < z && z < x {
            return [y, z, x];
        };
        if y < x && x < z {
            return [y, x, z];
        };
        if z < y && y < x {
            return [z, y, x];
        };
        if z < x && x < y {
            return [z, x, y];
        };
        panic!("This shouldn't happen");
    };

    for line in ele_lines {
        let line = line.unwrap();
        let mut words = line.split_whitespace();
        words.next();
        let v1 = words.next().unwrap().parse::<usize>().unwrap();
        let v2 = words.next().unwrap().parse::<usize>().unwrap();
        let v3 = words.next().unwrap().parse::<usize>().unwrap();
        let v4 = words.next().unwrap().parse::<usize>().unwrap();
        prim_connected_vert_indices.push([v1, v2, v3, v4]);
        // don't know why blender don't render object double sided
        let f1 = sort((v1, v2, v3));
        let f2 = sort((v1, v2, v4));
        let f3 = sort((v1, v3, v4));
        let f4 = sort((v2, v3, v4));

        // this order can correctly rendered by blender
        // let f1 = [v1, v2, v3];
        // let f2 = [v1,v4,v2];
        // let f3 = [v1,v3,v4];
        // let f4 = [v2,v4,v3];
        if surface.contains(&f1) {
            surface.remove(&f1);
        } else {
            surface.insert(f1);
        }
        if surface.contains(&f2) {
            surface.remove(&f2);
        } else {
            surface.insert(f2);
        }
        if surface.contains(&f3) {
            surface.remove(&f3);
        } else {
            surface.insert(f3);
        }
        if surface.contains(&f4) {
            surface.remove(&f4);
        } else {
            surface.insert(f4);
        }

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
        surface: Some(surface),
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
