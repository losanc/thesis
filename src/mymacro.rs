#[macro_export]
macro_rules! mylog {
    ($log:expr, $infor:tt, $value:expr) => {
        #[cfg(feature = "log")]
        {
            writeln!($log, "{} : {}", $infor, $value).unwrap();
        }
    };
}
