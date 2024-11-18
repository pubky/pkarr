use std::sync::Arc;

use rustls::client::danger::ServerCertVerifier;

use crate::Client;

#[derive(Debug)]
struct CertVerifier;

impl ServerCertVerifier for CertVerifier {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        vec![]
    }
}

impl From<Client> for CertVerifier {
    fn from(client: Client) -> Self {
        CertVerifier
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "relay"))]
impl From<crate::client::relay::Client> for CertVerifier {
    fn from(client: crate::client::relay::Client) -> Self {
        CertVerifier
    }
}

impl From<Client> for rustls::ClientConfig {
    fn from(client: Client) -> Self {
        let verifier: CertVerifier = client.into();

        rustls::ClientConfig::builder()
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(verifier))
            .with_no_client_auth()
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "relay"))]
impl From<crate::client::relay::Client> for rustls::ClientConfig {
    fn from(client: crate::client::relay::Client) -> Self {
        let verifier: CertVerifier = client.into();

        rustls::ClientConfig::builder()
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(verifier))
            .with_no_client_auth()
    }
}
