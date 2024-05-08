use futures::future::select_ok;
use std::str;

use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::RequestMode;

use crate::{Error, PublicKey, Result, SignedPacket, DEFAULT_RELAYS};

use tracing::debug;

#[derive(Debug, Clone)]
pub struct PkarrRelayClient {
    relays: Vec<String>,
}

impl Default for PkarrRelayClient {
    fn default() -> Self {
        Self::new(DEFAULT_RELAYS.map(|s| s.into()).to_vec()).unwrap()
    }
}

impl PkarrRelayClient {
    pub fn new(relays: Vec<String>) -> Result<Self> {
        if relays.is_empty() {
            return Err(Error::EmptyListOfRelays);
        }

        Ok(Self { relays })
    }

    /// Publishes a [SignedPacket] to this client's relays.
    ///
    /// Return the first successful completion, or the last failure.
    ///
    /// # Errors
    /// - Returns [Error::WasmRelayError] For Error responses
    /// - Returns [Error::JsError] If an error happened on JS side.
    pub async fn publish(&self, signed_packet: &SignedPacket) -> Result<()> {
        let futures = self.relays.iter().map(|relay| {
            Box::pin(async move {
                let url = format!("{relay}/{}", signed_packet.public_key());
                publish_inner(&url, signed_packet.to_relay_payload().to_vec()).await
            })
        });

        match select_ok(futures).await {
            Ok((response, _)) => Ok(response),
            Err(e) => Err(e),
        }
    }

    /// Resolve a [SignedPacket] from this client's relays.
    ///
    /// Return the first successful response, or the failure from the last responding relay.
    ///
    /// # Errors
    ///
    /// - Returns [Error::WasmRelayError] For Error responses
    /// - Returns [Error::JsError] If an error happened on JS side.
    pub async fn resolve(&self, public_key: &PublicKey) -> Result<Option<SignedPacket>> {
        let futures = self.relays.iter().map(|relay| {
            Box::pin(async move {
                let url = format!("{relay}/{public_key}");

                match resolve_inner(&url).await {
                    Ok(bytes) => {
                        match SignedPacket::from_relay_payload(public_key, &bytes.into()) {
                            Ok(signed_packet) => {
                                return Ok(Some(signed_packet));
                            }
                            Err(error) => {
                                debug!(?url, ?error, "Invalid signed_packet");
                                return Err(error);
                            }
                        }
                    }
                    Err(error) => {
                        debug!(?url, ?error, "Error response");
                        return Err(error);
                    }
                }
            })
        });

        match select_ok(futures).await {
            Ok((response, _)) => Ok(response),
            Err(e) => Err(e),
        }
    }
}

async fn publish_inner(url: &String, bytes: Vec<u8>) -> Result<()> {
    let response = fetch_base(url, "PUT", Some(bytes)).await?;
    let bytes = response_body(&response).await?;

    if !response.ok() {
        return Err(Error::WasmRelayError(
            response.status(),
            str::from_utf8(&bytes).ok().unwrap_or("").to_string(),
        ));
    }

    Ok(())
}

async fn resolve_inner(url: &String) -> Result<Vec<u8>> {
    let response = fetch_base(url, "GET", None).await?;
    let bytes = response_body(&response).await?;

    if !response.ok() {
        return Err(Error::WasmRelayError(
            response.status(),
            str::from_utf8(&bytes).ok().unwrap_or("").to_string(),
        ));
    }

    Ok(bytes)
}

async fn response_body(response: &web_sys::Response) -> Result<Vec<u8>> {
    let array_buffer = JsFuture::from(
        response
            .array_buffer()
            .map_err(|error| Error::JsError(error))?,
    )
    .await
    .map_err(|error| Error::JsError(error))?;

    let uint8_array = js_sys::Uint8Array::new(&array_buffer);

    Ok(uint8_array.to_vec())
}

async fn fetch_base(
    url: &String,
    method: &str,
    body: Option<Vec<u8>>,
) -> Result<web_sys::Response> {
    let mut opts = web_sys::RequestInit::new();
    opts.method(method);
    opts.mode(RequestMode::Cors);

    if let Some(body) = body {
        let body_bytes: &[u8] = &body;
        let body_array: js_sys::Uint8Array = body_bytes.into();
        let js_value: &JsValue = body_array.as_ref();
        opts.body(Some(js_value));
    }

    let js_request = web_sys::Request::new_with_str_and_init(url, &opts)
        .map_err(|error| Error::JsError(error))?;

    let window = web_sys::window().unwrap();
    let response = JsFuture::from(window.fetch_with_request(&js_request))
        .await
        .map_err(|error| Error::JsError(error))?;

    let response: web_sys::Response = response.dyn_into().map_err(|error| Error::JsError(error))?;

    Ok(response)
}

#[cfg(test)]
mod tests {
    use wasm_bindgen_test::wasm_bindgen_test;

    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    use crate::{dns, Keypair, PkarrRelayClient, SignedPacket};

    #[macro_export]
    macro_rules! log {
        ($($arg:expr),*) => {
            web_sys::console::debug_1(&format!($($arg),*).into());
        };
    }

    #[wasm_bindgen_test]
    async fn basic() {
        let keypair = Keypair::random();

        let mut packet = dns::Packet::new_reply(0);
        packet.answers.push(dns::ResourceRecord::new(
            dns::Name::new("foo").unwrap(),
            dns::CLASS::IN,
            30,
            dns::rdata::RData::TXT("bar".try_into().unwrap()),
        ));

        let signed_packet = SignedPacket::from_packet(&keypair, &packet).unwrap();

        let path = format!("/{}", signed_packet.public_key());

        let client = PkarrRelayClient::new(vec![
            "http://fail.non".to_string(),
            "https://relay.pkarr.org".to_string(),
        ])
        .unwrap();

        client.publish(&signed_packet).await.unwrap();

        let resolved = client
            .resolve(&keypair.public_key())
            .await
            .unwrap()
            .unwrap();

        log!("{:?}", resolved);

        assert_eq!(resolved.as_bytes(), signed_packet.as_bytes());
    }
}
