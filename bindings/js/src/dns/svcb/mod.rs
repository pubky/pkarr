pub mod constants;
pub mod input;
pub mod output;
pub mod utils;

// Re-export main functionality for easier imports
pub use input::apply_svcb_params;
pub use output::to_js_object;
