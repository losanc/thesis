use crate::area;
use crate::Mesh;
use nalgebra::DVector;
use nalgebra::{matrix, SMatrix};
use rand::Rng;

fn volume_mass_construct(
    density: f64,
    verts: &DVector<f64>,
    prim_connected_vert_indices: &[[usize; 3]],
) -> (DVector<f64>, Vec<f64>, Vec<SMatrix<f64, 2, 2>>) {
    let n_prims = prim_connected_vert_indices.len();

    let mut mass = DVector::<f64>::zeros(verts.len());
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
    (mass, volumes, ma_invs)
}

pub fn plane(
    r: usize,
    c: usize,
    w: Option<f64>,
    h: Option<f64>,
    d: Option<f64>,
    uniform: bool,
) -> Mesh<2, 3> {
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

    let get_index = |r_ind: usize, c_ind: usize| -> usize { r_ind * r + c_ind };
    let w = w.unwrap_or(1.0);
    let h = h.unwrap_or(1.0);

    let mut rng = rand::thread_rng();
    let mut row_cooridantes: Vec<f64>;
    let mut col_cooridantes: Vec<f64>;
    if !uniform {
        row_cooridantes = (0..r - 1)
            .map(|_| ((r - 1) as f64) * w * rng.gen::<f64>())
            .collect();
        col_cooridantes = (0..c - 1)
            .map(|_| ((c - 1) as f64) * h * rng.gen::<f64>())
            .collect();
        row_cooridantes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        col_cooridantes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        row_cooridantes[0] = 0.0;
        col_cooridantes[0] = 0.0;
        row_cooridantes.push((r as f64 - 1.0) * w);
        col_cooridantes.push((c as f64 - 1.0) * h);
    } else {
        row_cooridantes = (0..r).map(|i| (i as f64) * w).collect();
        col_cooridantes = (0..c).map(|i| (i as f64) * h).collect();
    }
    let verts = DVector::from_fn(2 * r * c, |i, _| {
        if i % 2 == 1 {
            // (((i - 1) / 2) / r) as f64 * w
            col_cooridantes[(((i - 1) / 2) / r)]
        } else {
            row_cooridantes[((i / 2) % r)]
            // ((i / 2) % r) as f64 * h
        }
    });

    let mut prim_connected_vert_indices = Vec::<[usize; 3]>::new();
    let mut vert_connected_prim_indices = vec![Vec::<usize>::new(); r * c];
    prim_connected_vert_indices.reserve_exact(2 * (r - 1) * (c - 1));
    let mut count = 0;
    for i in 0..c - 1 {
        for j in 0..r - 1 {
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

    for i in 1..c {
        for j in 1..r {
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

        verts,
        velos: DVector::<f64>::zeros(2 * r * c),
        accls: DVector::<f64>::zeros(2 * r * c),
        masss: mass,

        volumes,
        ma_invs,

        prim_connected_vert_indices,
        vert_connected_prim_indices,
    }
}

pub fn fan(r: f64, s: usize, d: Option<f64>) -> Mesh<2, 3> {
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
            r * angle.sin()
        } else {
            let ind = (i / 2) as f64;
            let ratio = ind / s as f64;
            let angle = ratio * 2.0 * std::f64::consts::PI;
            r * angle.cos()
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
    vert_connected_prim_indices[0].push(s - 1);
    vert_connected_prim_indices[1].push(s - 1);
    vert_connected_prim_indices[s].push(s - 1);

    let (mass, volumes, ma_invs) =
        volume_mass_construct(density, &verts, &prim_connected_vert_indices);
    Mesh {
        n_verts,
        n_prims,

        verts,
        velos: DVector::<f64>::zeros(2 * n_verts),
        accls: DVector::<f64>::zeros(2 * n_verts),
        masss: mass,

        volumes,
        ma_invs,

        prim_connected_vert_indices,
        vert_connected_prim_indices,
    }
}

pub fn circle(r: f64, res: usize, d: Option<f64>) -> Mesh<2, 3> {
    // modifed from https://stackoverflow.com/questions/53406534/procedural-circle-mesh-with-uniform-faces
    // zhi you yi ju niu bi
    let dim = r / res as f64;
    let n_verts = res * (res + 1) * 3 + 1;
    let density = d.unwrap_or(1e3);
    let mut vertices = DVector::zeros(n_verts * 2);

    let mut count = 1;
    for circ in 0..res {
        let angle_step = (std::f64::consts::PI * 2.0) / ((circ as f64 + 1.0) * 6.0);
        for point in 0..(circ + 1) * 6 {
            let angle = angle_step * point as f64;
            vertices[2 * count] = angle.cos() * dim * (circ + 1) as f64;
            vertices[2 * count + 1] = angle.sin() * dim * (circ + 1) as f64;
            count += 1;
        }
    }

    let get_point_index = |c: usize, x: usize| -> usize {
        if c == 0 {
            return 0; // In case of center point
        }
        let c = c - 1;
        let x = x % ((c + 1) * 6); // Make the point index circular
                                   // Explanation: index = number of points in previous circles + central point + x
                                   // hence: (0+1+2+...+c)*6+x+1 = ((c/2)*(c+1))*6+x+1 = 3*c*(c+1)+x+1

        3 * c * (c + 1) + x + 1
    };
    let mut n_prims = 0;
    let mut prim_connected_vert_indices = Vec::<[usize; 3]>::new();
    let mut vert_connected_prim_indices = vec![Vec::<usize>::new(); count];

    for circ in 0..res {
        let mut other = 0;
        for point in 0..(circ + 1) * 6 {
            if point % (circ + 1) != 0 {
                let v1 = get_point_index(circ, other + 1);
                let v2 = get_point_index(circ, other);
                let v3 = get_point_index(circ + 1, point);

                prim_connected_vert_indices.push([v1, v2, v3]);
                vert_connected_prim_indices[v1].push(n_prims);
                vert_connected_prim_indices[v2].push(n_prims);
                vert_connected_prim_indices[v3].push(n_prims);
                n_prims += 1;

                let v1 = get_point_index(circ + 1, point);
                let v2 = get_point_index(circ + 1, point + 1);
                let v3 = get_point_index(circ, other + 1);

                prim_connected_vert_indices.push([v1, v2, v3]);
                vert_connected_prim_indices[v1].push(n_prims);
                vert_connected_prim_indices[v2].push(n_prims);
                vert_connected_prim_indices[v3].push(n_prims);

                other += 1;
                n_prims += 1;
            } else {
                let v1 = get_point_index(circ + 1, point);
                let v2 = get_point_index(circ + 1, point + 1);
                let v3 = get_point_index(circ, other);
                prim_connected_vert_indices.push([v1, v2, v3]);
                vert_connected_prim_indices[v1].push(n_prims);
                vert_connected_prim_indices[v2].push(n_prims);
                vert_connected_prim_indices[v3].push(n_prims);

                n_prims += 1;
            }
        }
    }
    let (mass, volumes, ma_invs) =
        volume_mass_construct(density, &vertices, &prim_connected_vert_indices);

    Mesh {
        n_verts,
        n_prims,

        verts: vertices,
        velos: DVector::<f64>::zeros(2 * n_verts),
        accls: DVector::<f64>::zeros(2 * n_verts),
        masss: mass,

        volumes,
        ma_invs,

        prim_connected_vert_indices,
        vert_connected_prim_indices,
    }
}
