use super::*;

/// WASM-compatible wrapper for SignedPacketBuilder
#[wasm_bindgen]
pub struct SignedPacketBuilder {
    inner: NativeSignedPacketBuilder,
}

#[wasm_bindgen]
impl SignedPacketBuilder {
    /// Create a new SignedPacketBuilder for WASM
    #[wasm_bindgen(constructor)]
    pub fn new() -> SignedPacketBuilder {
        SignedPacketBuilder {
            inner: NativeSignedPacketBuilder::default(),
        }
    }

    /// Add a TXT record to the packet
    ///
    /// # Arguments
    /// * `name` - The domain name (e.g., "example" or "subdomain.example")
    /// * `text` - The text content
    /// * `ttl` - Time to live in seconds
    #[wasm_bindgen(js_name = "addTxtRecord")]
    pub fn add_txt_record(&mut self, name: &str, text: &str, ttl: u32) -> Result<(), JsValue> {
        let name = Name::new_unchecked(name);
        let txt = TXT::new()
            .with_string(text)
            .map_err(|e| JsValue::from_str(&format!("Invalid TXT record: {}", e)))?;

        let record = ResourceRecord::new(name, CLASS::IN, ttl, RData::TXT(txt));
        self.inner = self.inner.clone().record(record);
        Ok(())
    }

    /// Add an A record (IPv4 address) to the packet
    ///
    /// # Arguments
    /// * `name` - The domain name
    /// * `address` - The IPv4 address as a string (e.g., "192.168.1.1")
    /// * `ttl` - Time to live in seconds
    #[wasm_bindgen(js_name = "addARecord")]
    pub fn add_a_record(&mut self, name: &str, address: &str, ttl: u32) -> Result<(), JsValue> {
        let addr: Ipv4Addr = address
            .parse()
            .map_err(|e| JsValue::from_str(&format!("Invalid IPv4 address: {}", e)))?;

        self.inner = self.inner.clone().a(Name::new_unchecked(name), addr, ttl);
        Ok(())
    }

    /// Add an AAAA record (IPv6 address) to the packet
    ///
    /// # Arguments
    /// * `name` - The domain name
    /// * `address` - The IPv6 address as a string (e.g., "::1")
    /// * `ttl` - Time to live in seconds
    #[wasm_bindgen(js_name = "addAAAARecord")]
    pub fn add_aaaa_record(&mut self, name: &str, address: &str, ttl: u32) -> Result<(), JsValue> {
        let addr: Ipv6Addr = address
            .parse()
            .map_err(|e| JsValue::from_str(&format!("Invalid IPv6 address: {}", e)))?;

        self.inner = self
            .inner
            .clone()
            .aaaa(Name::new_unchecked(name), addr, ttl);
        Ok(())
    }

    /// Add a CNAME record to the packet
    ///
    /// # Arguments
    /// * `name` - The domain name
    /// * `target` - The target domain name
    /// * `ttl` - Time to live in seconds
    #[wasm_bindgen(js_name = "addCnameRecord")]
    pub fn add_cname_record(&mut self, name: &str, target: &str, ttl: u32) -> Result<(), JsValue> {
        self.inner =
            self.inner
                .clone()
                .cname(Name::new_unchecked(name), Name::new_unchecked(target), ttl);
        Ok(())
    }

    /// Add an HTTPS record to the packet
    ///
    /// # Arguments
    /// * `name` - The domain name
    /// * `priority` - Service priority (0-65535)
    /// * `target` - The target server domain name
    /// * `ttl` - Time to live in seconds
    #[wasm_bindgen(js_name = "addHttpsRecord")]
    pub fn add_https_record(
        &mut self,
        name: &str,
        priority: u16,
        target: &str,
        ttl: u32,
    ) -> Result<(), JsValue> {
        let svcb = SVCB::new(priority, Name::new_unchecked(target));
        self.inner = self
            .inner
            .clone()
            .https(Name::new_unchecked(name), svcb, ttl);
        Ok(())
    }

    /// Add an SVCB (Service Binding) record to the packet
    ///
    /// # Arguments
    /// * `name` - The domain name
    /// * `priority` - Service priority (0-65535)
    /// * `target` - The target server domain name
    /// * `ttl` - Time to live in seconds
    #[wasm_bindgen(js_name = "addSvcbRecord")]
    pub fn add_svcb_record(
        &mut self,
        name: &str,
        priority: u16,
        target: &str,
        ttl: u32,
    ) -> Result<(), JsValue> {
        let svcb = SVCB::new(priority, Name::new_unchecked(target));
        self.inner = self
            .inner
            .clone()
            .svcb(Name::new_unchecked(name), svcb, ttl);
        Ok(())
    }

    /// Add an NS record to the packet
    ///
    /// # Arguments
    /// * `name` - The domain name
    /// * `nameserver` - The nameserver domain name
    /// * `ttl` - Time to live in seconds
    #[wasm_bindgen(js_name = "addNsRecord")]
    pub fn add_ns_record(&mut self, name: &str, nameserver: &str, ttl: u32) -> Result<(), JsValue> {
        use simple_dns::rdata::NS;
        let ns = NS(Name::new_unchecked(nameserver));
        self.inner = self.inner.clone().ns(Name::new_unchecked(name), ns, ttl);
        Ok(())
    }

    /// Set the timestamp for the packet (optional)
    ///
    /// # Arguments
    /// * `timestamp_ms` - Timestamp in milliseconds since Unix epoch
    #[wasm_bindgen(js_name = "setTimestamp")]
    pub fn set_timestamp(&mut self, timestamp_ms: f64) {
        self.inner = self
            .inner
            .clone()
            .timestamp(Timestamp::from(timestamp_ms as u64 * 1000)); // Convert ms to microseconds
    }

    /// Build and sign the packet with the given keypair
    ///
    /// # Arguments
    /// * `keypair` - The Keypair to sign with
    ///
    /// # Returns
    /// * `SignedPacket` - The signed packet ready for publishing
    #[wasm_bindgen(js_name = "buildAndSign")]
    pub fn build_and_sign(&self, keypair: &super::Keypair) -> Result<super::SignedPacket, JsValue> {
        let signed_packet = self
            .inner
            .clone()
            .sign(&keypair.keypair)
            .map_err(|e| JsValue::from_str(&format!("Failed to build packet: {}", e)))?;

        Ok(super::SignedPacket::from(signed_packet))
    }

    /// Clear all records from the builder
    #[wasm_bindgen(js_name = "clear")]
    pub fn clear(&mut self) {
        self.inner = NativeSignedPacketBuilder::default();
    }

    /// Create a new builder instance (static method)
    #[wasm_bindgen(js_name = "builder")]
    pub fn builder() -> SignedPacketBuilder {
        SignedPacketBuilder::new()
    }
}
