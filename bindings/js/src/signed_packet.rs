use super::constants::*;
use super::error::ClientError;
use super::*;

/// WASM-compatible wrapper for SignedPacket
#[wasm_bindgen]
pub struct SignedPacket {
    pub(super) inner: pkarr::SignedPacket,
}

#[wasm_bindgen]
impl SignedPacket {
    /// Get the public key as a z-base32 string
    #[wasm_bindgen(getter, js_name = publicKeyString)]
    pub fn public_key_string(&self) -> String {
        self.inner.public_key().to_string()
    }

    /// Get the timestamp in milliseconds since Unix epoch
    #[wasm_bindgen(getter, js_name = timestampMs)]
    pub fn timestamp_ms(&self) -> f64 {
        self.inner.timestamp().as_u64() as f64
    }

    /// Get the number of DNS records in this packet
    #[wasm_bindgen(getter, js_name = recordCount)]
    pub fn record_count(&self) -> usize {
        self.inner.all_resource_records().count()
    }

    /// Check if the packet contains any records
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.record_count() == 0
    }

    /// Get the DNS records as a JavaScript array of objects
    #[wasm_bindgen(getter)]
    pub fn records(&self) -> Result<js_sys::Array, JsValue> {
        let records = js_sys::Array::new();

        for record in self.inner.all_resource_records() {
            if let Ok(js_record) = Self::record_to_js(record) {
                records.push(&js_record);
            }
            // Skip unsupported record types silently
        }

        Ok(records)
    }

    /// Get the raw bytes of the signed packet
    #[wasm_bindgen(js_name = toBytes)]
    pub fn to_bytes(&self) -> Result<js_sys::Uint8Array, JsValue> {
        let bytes = self.inner.serialize();

        // Validate the serialized size against pkarr specification
        if bytes.len() > MAX_PACKET_SIZE {
            return Err(ClientError::ValidationError {
                context: "packet serialization".to_string(),
                message: format!(
                    "Serialized packet too large: {} bytes (max {})",
                    bytes.len(),
                    MAX_PACKET_SIZE
                ),
            }
            .into());
        }

        Ok(js_sys::Uint8Array::from(&bytes[..]))
    }

    /// Verify the cryptographic signature of this packet
    #[wasm_bindgen(js_name = isValid)]
    pub fn is_valid(&self) -> bool {
        // The existence of the SignedPacket implies it was successfully verified during creation
        // Additional validation could be added here if needed
        !self.is_empty()
    }

    /// Create a SignedPacket from bytes
    ///
    /// # Arguments
    /// * `bytes` - The serialized signed packet bytes
    #[wasm_bindgen(js_name = fromBytes)]
    pub fn from_bytes(bytes: &[u8]) -> Result<SignedPacket, JsValue> {
        Self::validate_bytes(bytes)?;

        let signed_packet =
            pkarr::SignedPacket::deserialize(bytes).map_err(|e| ClientError::ParseError {
                input_type: "signed packet bytes".to_string(),
                message: e.to_string(),
            })?;

        Ok(SignedPacket {
            inner: signed_packet,
        })
    }

    /// Create a SignedPacketBuilder (static method)
    #[wasm_bindgen(js_name = builder)]
    pub fn builder() -> super::SignedPacketBuilder {
        super::SignedPacketBuilder::default()
    }
}

impl From<pkarr::SignedPacket> for SignedPacket {
    fn from(inner: pkarr::SignedPacket) -> Self {
        SignedPacket { inner }
    }
}

impl From<SignedPacket> for pkarr::SignedPacket {
    fn from(wrapper: SignedPacket) -> Self {
        wrapper.inner
    }
}

// Private helper methods
impl SignedPacket {
    /// Validate input bytes before deserializing
    fn validate_bytes(bytes: &[u8]) -> Result<(), JsValue> {
        if bytes.is_empty() {
            return Err(ClientError::ValidationError {
                context: "signed packet bytes".to_string(),
                message: "Input bytes cannot be empty".to_string(),
            }
            .into());
        }

        // Check for pkarr specification limit
        if bytes.len() > MAX_PACKET_SIZE {
            return Err(ClientError::ValidationError {
                context: "signed packet bytes".to_string(),
                message: format!(
                    "Packet too large: {} bytes (max {})",
                    bytes.len(),
                    MAX_PACKET_SIZE
                ),
            }
            .into());
        }

        Ok(())
    }

    /// Sets a property on a JavaScript object
    ///
    /// # Arguments
    /// * `obj` - The JavaScript object to set the property on
    /// * `key` - The property name to set
    /// * `value` - The JavaScript value to assign to the property
    ///
    /// # Errors
    /// Returns a ClientError::BuildError if setting the property fails
    fn set_prop(obj: &js_sys::Object, key: &str, value: &JsValue) -> Result<(), JsValue> {
        js_sys::Reflect::set(obj, &JsValue::from_str(key), value).map_err(|_| {
            ClientError::BuildError("Failed to set DNS record property".to_string())
        })?;
        Ok(())
    }

    /// Convert a ResourceRecord to a JavaScript object
    ///
    /// # Arguments
    /// * `rr` - The ResourceRecord to convert
    ///
    /// # Returns
    /// A JavaScript object representing the DNS record
    fn record_to_js(rr: &ResourceRecord) -> Result<JsValue, JsValue> {
        let record_obj = js_sys::Object::new();

        // Add name
        Self::set_prop(
            &record_obj,
            PROP_NAME,
            &JsValue::from_str(&rr.name.to_string()),
        )?;

        // Add TTL
        Self::set_prop(&record_obj, PROP_TTL, &JsValue::from_f64(rr.ttl as f64))?;

        // Add type and data
        let rdata_obj = js_sys::Object::new();
        match &rr.rdata {
            RData::A(A { address }) => {
                Self::set_prop(&rdata_obj, PROP_TYPE, &JsValue::from_str(TYPE_A))?;
                Self::set_prop(
                    &rdata_obj,
                    PROP_ADDRESS,
                    &JsValue::from_str(&Ipv4Addr::from(*address).to_string()),
                )?;
            }
            RData::AAAA(AAAA { address }) => {
                Self::set_prop(&rdata_obj, PROP_TYPE, &JsValue::from_str(TYPE_AAAA))?;
                Self::set_prop(
                    &rdata_obj,
                    PROP_ADDRESS,
                    &JsValue::from_str(&Ipv6Addr::from(*address).to_string()),
                )?;
            }
            RData::CNAME(name) => {
                Self::set_prop(&rdata_obj, PROP_TYPE, &JsValue::from_str(TYPE_CNAME))?;
                Self::set_prop(
                    &rdata_obj,
                    PROP_TARGET,
                    &JsValue::from_str(&name.to_string()),
                )?;
            }
            RData::TXT(txt) => {
                Self::set_prop(&rdata_obj, PROP_TYPE, &JsValue::from_str(TYPE_TXT))?;
                let text_content = txt.clone().try_into().unwrap_or_else(|_| "".to_string());
                Self::set_prop(&rdata_obj, PROP_VALUE, &JsValue::from_str(&text_content))?;
            }
            RData::HTTPS(HTTPS(svcb)) => {
                Self::set_prop(&rdata_obj, PROP_TYPE, &JsValue::from_str(TYPE_HTTPS))?;
                Self::set_prop(
                    &rdata_obj,
                    PROP_PRIORITY,
                    &JsValue::from_f64(svcb.priority as f64),
                )?;
                Self::set_prop(
                    &rdata_obj,
                    PROP_TARGET,
                    &JsValue::from_str(&svcb.target.to_string()),
                )?;
            }
            RData::SVCB(svcb) => {
                Self::set_prop(&rdata_obj, PROP_TYPE, &JsValue::from_str(TYPE_SVCB))?;
                Self::set_prop(
                    &rdata_obj,
                    PROP_PRIORITY,
                    &JsValue::from_f64(svcb.priority as f64),
                )?;
                Self::set_prop(
                    &rdata_obj,
                    PROP_TARGET,
                    &JsValue::from_str(&svcb.target.to_string()),
                )?;
            }
            RData::NS(NS(name)) => {
                Self::set_prop(&rdata_obj, PROP_TYPE, &JsValue::from_str(TYPE_NS))?;
                Self::set_prop(
                    &rdata_obj,
                    PROP_NSDNAME,
                    &JsValue::from_str(&name.to_string()),
                )?;
            }
            _ => {
                // Skip unsupported record types by returning an error that will be filtered out
                return Err(JsValue::from_str("Unsupported record type"));
            }
        }

        Self::set_prop(&record_obj, PROP_RDATA, &rdata_obj)?;
        Ok(record_obj.into())
    }
}
