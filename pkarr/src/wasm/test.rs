use crate::wasm::{Keypair, SignedPacketBuilder, Utils};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_types_instantiate() {
        // Test that our WASM types can be created
        let builder = SignedPacketBuilder::new();
        let keypair = Keypair::random();

        assert!(!keypair.public_key_string().is_empty());
    }

    #[test]
    fn test_utils_methods() {
        // Test that utils methods are accessible
        let valid_key = Utils::validate_public_key("invalid");
        assert!(!valid_key);

        let relays = Utils::default_relays();
        assert!(relays.length() > 0);
    }
}
