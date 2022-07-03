/// calculates the area of triangle
#[inline]
pub fn area(x1: f64, y1: f64, x2: f64, y2: f64, x3: f64, y3: f64) -> f64 {
    0.5 * ((x1 - x3) * (y2 - y1) - (x1 - x2) * (y3 - y1)).abs()
}

/// calcuates the volume of tet
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn volume(
    x1: f64,
    y1: f64,
    z1: f64,
    x2: f64,
    y2: f64,
    z2: f64,
    x3: f64,
    y3: f64,
    z3: f64,
    x4: f64,
    y4: f64,
    z4: f64,
) -> f64 {
    ((x4 - x1) * ((y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1))
        + (y4 - y1) * ((z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1))
        + (z4 - z1) * ((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))).abs()
        / 6.0
}
