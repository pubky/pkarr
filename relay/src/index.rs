use axum::response::Html;

use crate::AppState;

const TEMPLATE: &str = include_str!("index.html");

/// Render the relay index page.
pub(crate) async fn render(state: &AppState) -> Html<String> {
    let cache = state.dht.cache_stats();

    let size = cache.size;
    let capacity = cache.capacity;
    let utilization = 100.0 * size as f32 / capacity as f32;

    let info = state.dht.info().await;

    let (dht_size, confidence) = info.dht_size_estimate();
    let node_port = info
        .public_address()
        .map(|addr| addr.port())
        .unwrap_or(info.local_addr().port());
    let html_content = TEMPLATE
        .replace("__VERSION__", env!("CARGO_PKG_VERSION"))
        .replace("__CACHE_SIZE__", &format_number(size))
        .replace("__CACHE_CAPACITY__", &format_number(capacity))
        .replace("__CACHE_UTILIZATION__", &format!("{utilization:.2}"))
        .replace("__NODE_PORT__", &node_port.to_string())
        .replace("__NODE_FIREWALLED__", &info.firewalled().to_string())
        .replace("__DHT_SIZE__", &format_number(dht_size))
        .replace("__DHT_CONFIDENCE__", &format!("{confidence:.2}"));

    Html(html_content)
}

fn format_number(num: usize) -> String {
    if num >= 1_000_000_000 {
        return format!("{:.1}B", num as f64 / 1_000_000_000.0);
    } else if num >= 1_000_000 {
        return format!("{:.1}M", num as f64 / 1_000_000.0);
    } else if num >= 1_000 {
        return format!("{:.1}K", num as f64 / 1_000.0);
    }

    let num_str = num.to_string();
    let mut result = String::new();
    let len = num_str.len();

    for (i, c) in num_str.chars().enumerate() {
        if i > 0 && (len - i) % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }

    result
}
