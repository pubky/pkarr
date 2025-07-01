use super::*;

/// WASM-compatible wrapper for SignedPacket
#[wasm_bindgen]
pub struct SignedPacket {
    pub(super) inner: NativeSignedPacket,
}

#[wasm_bindgen]
impl SignedPacket {
    /// Get the public key as a z-base32 string
    #[wasm_bindgen(getter, js_name = publicKeyString)]
    pub fn public_key_string(&self) -> String {
        self.inner.public_key().to_string()
    }

    /// Get the timestamp in milliseconds
    #[wasm_bindgen(getter, js_name = timestampMs)]
    pub fn timestamp_ms(&self) -> f64 {
        self.inner.timestamp().as_u64() as f64
    }

    /// Get the DNS records
    #[wasm_bindgen(getter)]
    pub fn records(&self) -> js_sys::Array {
        let records = js_sys::Array::new();

        for record in self.inner.all_resource_records() {
            let record_obj = js_sys::Object::new();

            // Add name
            js_sys::Reflect::set(
                &record_obj,
                &JsValue::from_str("name"),
                &JsValue::from_str(&record.name.to_string()),
            )
            .unwrap();

            // Add TTL
            js_sys::Reflect::set(
                &record_obj,
                &JsValue::from_str("ttl"),
                &JsValue::from_f64(record.ttl as f64),
            )
            .unwrap();

            // Add type and data
            let rdata_obj = js_sys::Object::new();
            match &record.rdata {
                RData::A(A { address }) => {
                    js_sys::Reflect::set(
                        &rdata_obj,
                        &JsValue::from_str("type"),
                        &JsValue::from_str("A"),
                    )
                    .unwrap();
                    js_sys::Reflect::set(
                        &rdata_obj,
                        &JsValue::from_str("address"),
                        &JsValue::from_str(&std::net::Ipv4Addr::from(*address).to_string()),
                    )
                    .unwrap();
                }
                RData::AAAA(AAAA { address }) => {
                    js_sys::Reflect::set(
                        &rdata_obj,
                        &JsValue::from_str("type"),
                        &JsValue::from_str("AAAA"),
                    )
                    .unwrap();
                    js_sys::Reflect::set(
                        &rdata_obj,
                        &JsValue::from_str("address"),
                        &JsValue::from_str(&std::net::Ipv6Addr::from(*address).to_string()),
                    )
                    .unwrap();
                }
                RData::CNAME(name) => {
                    js_sys::Reflect::set(
                        &rdata_obj,
                        &JsValue::from_str("type"),
                        &JsValue::from_str("CNAME"),
                    )
                    .unwrap();
                    js_sys::Reflect::set(
                        &rdata_obj,
                        &JsValue::from_str("target"),
                        &JsValue::from_str(&name.to_string()),
                    )
                    .unwrap();
                }
                RData::TXT(txt) => {
                    js_sys::Reflect::set(
                        &rdata_obj,
                        &JsValue::from_str("type"),
                        &JsValue::from_str("TXT"),
                    )
                    .unwrap();
                    let text_content = txt.clone().try_into().unwrap_or_else(|_| "".to_string());
                    js_sys::Reflect::set(
                        &rdata_obj,
                        &JsValue::from_str("value"),
                        &JsValue::from_str(&text_content),
                    )
                    .unwrap();
                }
                RData::HTTPS(HTTPS(svcb)) => {
                    js_sys::Reflect::set(
                        &rdata_obj,
                        &JsValue::from_str("type"),
                        &JsValue::from_str("HTTPS"),
                    )
                    .unwrap();

                    js_sys::Reflect::set(
                        &rdata_obj,
                        &JsValue::from_str("priority"),
                        &JsValue::from_f64(svcb.priority as f64),
                    )
                    .unwrap();

                    js_sys::Reflect::set(
                        &rdata_obj,
                        &JsValue::from_str("target"),
                        &JsValue::from_str(&svcb.target.to_string()),
                    )
                    .unwrap();
                }
                RData::SVCB(svcb) => {
                    js_sys::Reflect::set(
                        &rdata_obj,
                        &JsValue::from_str("type"),
                        &JsValue::from_str("SVCB"),
                    )
                    .unwrap();

                    js_sys::Reflect::set(
                        &rdata_obj,
                        &JsValue::from_str("priority"),
                        &JsValue::from_f64(svcb.priority as f64),
                    )
                    .unwrap();

                    js_sys::Reflect::set(
                        &rdata_obj,
                        &JsValue::from_str("target"),
                        &JsValue::from_str(&svcb.target.to_string()),
                    )
                    .unwrap();
                }
                RData::NS(NS(name)) => {
                    js_sys::Reflect::set(
                        &rdata_obj,
                        &JsValue::from_str("type"),
                        &JsValue::from_str("NS"),
                    )
                    .unwrap();

                    js_sys::Reflect::set(
                        &rdata_obj,
                        &JsValue::from_str("nsdname"),
                        &JsValue::from_str(&name.to_string()),
                    )
                    .unwrap();
                }
                _ => {
                    // Skip unsupported record types
                    continue;
                }
            }

            js_sys::Reflect::set(&record_obj, &JsValue::from_str("rdata"), &rdata_obj).unwrap();

            records.push(&record_obj);
        }

        records
    }

    /// Get the raw bytes of the signed packet
    #[wasm_bindgen(js_name = toBytes)]
    pub fn to_bytes(&self) -> js_sys::Uint8Array {
        js_sys::Uint8Array::from(&self.inner.serialize()[..])
    }

    /// Create a SignedPacket from bytes
    #[wasm_bindgen(js_name = fromBytes)]
    pub fn from_bytes(bytes: &[u8]) -> Result<SignedPacket, JsValue> {
        let signed_packet = NativeSignedPacket::deserialize(bytes)
            .map_err(|e| JsValue::from_str(&format!("Invalid signed packet: {}", e)))?;
        Ok(SignedPacket {
            inner: signed_packet,
        })
    }

    /// Create a SignedPacketBuilder (static method)
    #[wasm_bindgen(js_name = builder)]
    pub fn builder() -> super::SignedPacketBuilder {
        super::SignedPacketBuilder::new()
    }
}

impl From<NativeSignedPacket> for SignedPacket {
    fn from(inner: NativeSignedPacket) -> Self {
        SignedPacket { inner }
    }
}

impl From<SignedPacket> for NativeSignedPacket {
    fn from(wrapper: SignedPacket) -> Self {
        wrapper.inner
    }
}
