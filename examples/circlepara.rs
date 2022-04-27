use mesh::NeoHookean2d;

pub const E: f64 = 2e7;
pub const NU: f64 = 0.33;
pub const MIU: f64 = E / (2.0 * (1.0 + NU));
pub const LAMBDA: f64 = (E * NU) / ((1.0 + NU) * (1.0 - 2.0 * NU));
pub const DT: f64 = 1.0 / 60.0;
pub const DIM: usize = 2;
pub const DAMP: f64 = 1.0;
pub const RES: usize = 15;
pub const R: f64 = 5.0;
pub const TOTAL_FRAME: usize = 100;
pub type EnergyType = NeoHookean2d;

#[allow(dead_code)]
pub const CO_NUM: usize = DIM * (DIM + 1);

#[allow(dead_code)]
pub const ACTIVE_SET_EPI: f64 = 0.1;

#[allow(dead_code)]
fn main() {}
