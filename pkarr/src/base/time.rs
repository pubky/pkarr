#[cfg(not(target_arch = "wasm32"))]
use std::time::SystemTime;

#[cfg(not(target_arch = "wasm32"))]
/// Return the number of microseconds since [SystemTime::UNIX_EPOCH]
pub fn system_time() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .expect("time drift")
        .as_micros() as u64
}

#[cfg(target_arch = "wasm32")]
/// Return the number of microseconds since [SystemTime::UNIX_EPOCH]
pub fn system_time() -> u64 {
    // Won't be an issue for more than 5000 years!
    (js_sys::Date::now() as u64 )
    // Turn miliseconds to microseconds
    * 1000
}
