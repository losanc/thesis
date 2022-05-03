use mesh::NeoHookean3d;

pub const E: f64 = 1e5;
pub const NU: f64 = 0.33;
pub const MIU: f64 = E / (2.0 * (1.0 + NU));
pub const LAMBDA: f64 = (E * NU) / ((1.0 + NU) * (1.0 - 2.0 * NU));
pub const DT: f64 = 0.01;
pub const DIM: usize = 3;
pub const NFIXED_VERT: usize = 20;
pub const DAMP: f64 = 1.0;
pub const TOTAL_FRAME: usize = 100;

#[allow(dead_code)]
pub const CO_NUM: usize = DIM * (DIM + 1);

#[allow(dead_code)]
pub const ACTIVE_SET_EPI: f64 = 0.8;
pub type EnergyType = NeoHookean3d;

#[allow(dead_code)]
fn main() {}
