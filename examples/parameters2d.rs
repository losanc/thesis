use mesh::NeoHookean2d;

pub const E: f64 = 1e7;
pub const NU: f64 = 0.33;
pub const MIU: f64 = E / (2.0 * (1.0 + NU));
pub const LAMBDA: f64 = (E * NU) / ((1.0 + NU) * (1.0 - 2.0 * NU));
pub const DT: f64 = 1.0 / 60.0;
pub const DIM: usize = 2;
pub const NFIXED_VERT: usize = 20;
#[allow(non_upper_case_globals)]
pub const c: usize = 80;
pub const DAMP: f64 = 1.0;
pub const SIZE: f64 = 0.25;
pub const TOTAL_FRAME: usize = 30;
pub type EnergyType = NeoHookean2d;
pub const CO_NUM: usize = DIM * (DIM + 1);
pub const ACTIVE_SET_EPI: f64 = 0.01;
