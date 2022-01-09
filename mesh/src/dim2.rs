use crate::area;
use crate::Mesh;
use nalgebra::DVector;
use nalgebra::{matrix, SMatrix};

fn volume_mass_construct(
    density: f64,
    verts: &DVector<f64>,
    prim_connected_vert_indices: &Vec<[usize; 3]>,
) -> (DVector<f64>, Vec<f64>, Vec<SMatrix<f64, 2, 2>>) {
    let n_verts = verts.len();
    let n_prims = prim_connected_vert_indices.len();

    let mut mass = DVector::<f64>::zeros(2 * n_verts);
    let mut volumes = Vec::<f64>::new();
    let mut ma_invs = Vec::<SMatrix<f64, 2, 2>>::new();

    volumes.reserve_exact(n_prims);
    ma_invs.reserve_exact(n_prims);

    for [i, j, k] in prim_connected_vert_indices.iter() {
        let size = area(
            verts[*i * 2],
            verts[*i * 2 + 1],
            verts[*j * 2],
            verts[*j * 2 + 1],
            verts[*k * 2],
            verts[*k * 2 + 1],
        );

        mass[*i * 2] += 0.333 * size * density;
        mass[*i * 2 + 1] += 0.333 * size * density;
        mass[*j * 2] += 0.333 * size * density;
        mass[*j * 2 + 1] += 0.333 * size * density;
        mass[*k * 2] += 0.333 * size * density;
        mass[*k * 2 + 1] += 0.333 * size * density;

        volumes.push(size);
        let matrix = matrix![
            verts[*k * 2] - verts[*i * 2], verts[*j * 2] - verts[*i * 2];
            verts[*k * 2 + 1] - verts[*i * 2 + 1 ], verts[*j * 2 + 1 ] - verts[*i * 2 + 1];
        ];
        ma_invs.push(matrix.try_inverse().unwrap());
    }
    return (mass, volumes, ma_invs);
}

pub fn plane(r: usize, c: usize, d: Option<f64>) -> Mesh<2, 3> {
    assert!(r * c > 0);
    //  the shape of this Plane

    // r-1 ----------r-1*c-1
    //     .............
    //     |   |   |   |
    // 1   -------------
    //     c  c+1...
    //     |   |   |   |
    // 0   -------------
    //   0   1   2   c-1

    let get_index = |r_ind: usize, c_ind: usize| -> usize { r_ind * c + c_ind };

    // let mut vers = Vec::<SVector<f64, 2>>::new();
    let verts = DVector::from_fn(2 * r * c, |i, _| {
        if i % 2 == 1 {
            return (((i - 1) / 2) % c) as f64;
        } else {
            return ((i / 2) / c) as f64;
        }
    });

    let mut prim_connected_vert_indices = Vec::<[usize; 3]>::new();
    let mut vert_connected_prim_indices = vec![Vec::<usize>::new(); r * c];
    prim_connected_vert_indices.reserve_exact(2 * (r - 1) * (c - 1));
    let mut count = 0;
    for i in 0..r - 1 {
        for j in 0..c - 1 {
            let v1 = get_index(i, j);
            let v2 = get_index(i, j + 1);
            let v3 = get_index(i + 1, j);

            prim_connected_vert_indices.push([v1, v2, v3]);

            vert_connected_prim_indices[v1].push(count);
            vert_connected_prim_indices[v2].push(count);
            vert_connected_prim_indices[v3].push(count);
            count += 1;
        }
    }

    for i in 1..r {
        for j in 1..c {
            let v1 = get_index(i, j);
            let v2 = get_index(i, j - 1);
            let v3 = get_index(i - 1, j);

            prim_connected_vert_indices.push([v1, v2, v3]);

            vert_connected_prim_indices[v1].push(count);
            vert_connected_prim_indices[v2].push(count);
            vert_connected_prim_indices[v3].push(count);
            count += 1
        }
    }

    let density = d.unwrap_or(1e3);

    let (mass, volumes, ma_invs) =
        volume_mass_construct(density, &verts, &prim_connected_vert_indices);

    Mesh {
        n_verts: r * c,
        n_prims: count,

        verts: verts,
        velos: DVector::<f64>::zeros(2 * r * c),
        accls: DVector::<f64>::zeros(2 * r * c),
        masss: mass,

        volumes: volumes,
        ma_invs: ma_invs,

        prim_connected_vert_indices: prim_connected_vert_indices,
        vert_connected_prim_indices: vert_connected_prim_indices,
    }
}

pub fn circle(r: f64, s: usize, d: Option<f64>) -> Mesh<2, 3> {
    // the shape of this circle

    //            s/4
    //             |  ...     2
    //             |       /
    //             |   /
    // s/2---------0----------1
    //             |   \
    //             |       \
    //             |          s
    //           s*3/4
    //

    assert!(r > 0.0);
    assert!(s > 2);
    let n_verts = s + 1;
    let n_prims = s;

    let density = d.unwrap_or(1e3);

    let verts = DVector::from_fn(2 * (s + 1), |i, _| {
        if i <= 1 {
            return 0.0;
        }
        if i % 2 == 1 {
            let ind = ((i - 1) / 2) as f64;
            let ratio = ind / s as f64;
            let angle = ratio * 2.0 * std::f64::consts::PI;
            return r * angle.sin();
        } else {
            let ind = (i / 2) as f64;
            let ratio = ind / s as f64;
            let angle = ratio * 2.0 * std::f64::consts::PI;
            return r * angle.cos();
        }
    });

    let mut prim_connected_vert_indices = Vec::<[usize; 3]>::new();
    let mut vert_connected_prim_indices = vec![Vec::<usize>::new(); s + 1];
    prim_connected_vert_indices.reserve_exact(s);
    for i in 0..s - 1 {
        prim_connected_vert_indices.push([0, i + 1, i + 2]);
        vert_connected_prim_indices[0].push(i);
        vert_connected_prim_indices[i + 1].push(i);
        vert_connected_prim_indices[i + 2].push(i);
    }
    prim_connected_vert_indices.push([0, s, 1]);
    vert_connected_prim_indices[0].push(s);
    vert_connected_prim_indices[1].push(s);
    vert_connected_prim_indices[s].push(s);

    let (mass, volumes, ma_invs) =
        volume_mass_construct(density, &verts, &prim_connected_vert_indices);
    Mesh {
        n_verts: n_verts,
        n_prims: n_prims,

        verts: verts,
        velos: DVector::<f64>::zeros(2 * n_verts),
        accls: DVector::<f64>::zeros(2 * n_verts),
        masss: mass,

        volumes: volumes,
        ma_invs: ma_invs,

        prim_connected_vert_indices: prim_connected_vert_indices,
        vert_connected_prim_indices: vert_connected_prim_indices,
    }
}
