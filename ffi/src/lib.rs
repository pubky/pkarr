uniffi::setup_scaffolding!();
use pkarr::{PkarrClient, PublicKey};

#[uniffi::export]
fn say_hi() -> String {
    "Hello from Rust!".to_string()
}

#[uniffi::export]
async fn resolve(public_key: String) -> String {
    let client = PkarrClient::new();

    let str: &str = &public_key;
    let public_key: PublicKey = str.try_into().expect("Invalid zbase32 encoded key");
    
    if let Some(signed_packet) = client.resolve(public_key).await {
        return format!("{}", signed_packet);
    } else {
        return format!("Failed to resolve {}", str);
    }
}
