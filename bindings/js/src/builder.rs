use super::constants::*;
use super::error::ClientError;
use super::*;

/// WASM-compatible wrapper for SignedPacketBuilder
#[derive(Default)]
#[wasm_bindgen]
pub struct SignedPacketBuilder {
    inner: NativeSignedPacketBuilder,
}

#[wasm_bindgen]
impl SignedPacketBuilder {
    /// Create a new SignedPacketBuilder for WASM
    #[wasm_bindgen(constructor)]
    pub fn new() -> SignedPacketBuilder {
        Self::default()
    }

    /// Add a TXT record to the packet
    ///
    /// # Arguments
    /// * `name` - The domain name (e.g., "example" or "subdomain.example")
    /// * `text` - The text content
    /// * `ttl` - Time to live in seconds
    #[wasm_bindgen(js_name = "addTxtRecord")]
    pub fn add_txt_record(&mut self, name: &str, text: &str, ttl: u32) -> Result<(), JsValue> {
        let domain_name = self.parse_name(name)?;
        let txt = TXT::new()
            .with_string(text)
            .map_err(|e| ClientError::ValidationError {
                context: "TXT record".to_string(),
                message: e.to_string(),
            })?;

        let record = ResourceRecord::new(domain_name, CLASS::IN, ttl, RData::TXT(txt));
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
        let domain_name = self.parse_name(name)?;
        let addr: Ipv4Addr =
            address
                .parse()
                .map_err(|e: std::net::AddrParseError| ClientError::ParseError {
                    input_type: "IPv4 address".to_string(),
                    message: e.to_string(),
                })?;

        self.inner = self.inner.clone().a(domain_name, addr, ttl);
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
        let domain_name = self.parse_name(name)?;
        let addr: Ipv6Addr =
            address
                .parse()
                .map_err(|e: std::net::AddrParseError| ClientError::ParseError {
                    input_type: "IPv6 address".to_string(),
                    message: e.to_string(),
                })?;

        self.inner = self.inner.clone().aaaa(domain_name, addr, ttl);
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
        let (domain_name, target_name) = self.parse_name_pair(name, target)?;
        self.inner = self.inner.clone().cname(domain_name, target_name, ttl);
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
        let (domain_name, target_name) = self.parse_name_pair(name, target)?;
        let svcb = SVCB::new(priority, target_name);
        self.inner = self.inner.clone().https(domain_name, svcb, ttl);
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
        let (domain_name, target_name) = self.parse_name_pair(name, target)?;
        let svcb = SVCB::new(priority, target_name);
        self.inner = self.inner.clone().svcb(domain_name, svcb, ttl);
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
        let (domain_name, nameserver_name) = self.parse_name_pair(name, nameserver)?;
        let ns = simple_dns::rdata::NS(nameserver_name);
        self.inner = self.inner.clone().ns(domain_name, ns, ttl);
        Ok(())
    }

    /// Set the timestamp for the packet (optional)
    ///
    /// # Arguments
    /// * `timestamp_ms` - Timestamp in milliseconds since Unix epoch
    #[wasm_bindgen(js_name = "setTimestamp")]
    pub fn set_timestamp(&mut self, timestamp_ms: f64) {
        let timestamp_microseconds = (timestamp_ms as u64).saturating_mul(MS_TO_MICROSECONDS);
        self.inner = self
            .inner
            .clone()
            .timestamp(Timestamp::from(timestamp_microseconds));
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
            .map_err(|e| ClientError::BuildError(e.to_string()))?;

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
        Self::new()
    }
}

// Private helper methods
impl SignedPacketBuilder {
    /// Validate a domain name string
    fn validate_name(&self, name: &str) -> Result<(), JsValue> {
        if name.is_empty() {
            return Err(ClientError::ValidationError {
                context: "domain name".to_string(),
                message: "Name cannot be empty".to_string(),
            }
            .into());
        }

        // Check for obviously invalid characters
        if name.contains("..") || name.starts_with('-') || name.ends_with('-') {
            return Err(ClientError::ValidationError {
                context: "domain name".to_string(),
                message: "Invalid domain name format".to_string(),
            }
            .into());
        }

        Ok(())
    }

    /// Validate a single name and return a `Name`
    fn parse_name<'a>(&self, name: &'a str) -> Result<Name<'a>, JsValue> {
        self.validate_name(name)?;
        Ok(Name::new_unchecked(name))
    }

    /// Validate two domain names and return their `Name` representations
    fn parse_name_pair<'a>(
        &self,
        name: &'a str,
        target: &'a str,
    ) -> Result<(Name<'a>, Name<'a>), JsValue> {
        self.validate_name(name)?;
        self.validate_name(target)?;

        Ok((Name::new_unchecked(name), Name::new_unchecked(target)))
    }
}
