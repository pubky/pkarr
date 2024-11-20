use std::sync::Arc;

use rustls::{
    client::danger::{ServerCertVerified, ServerCertVerifier},
    pki_types::SubjectPublicKeyInfoDer,
    CertificateError,
};

use crate::{Client, PublicKey};

#[derive(Debug)]
struct CertVerifier;

impl ServerCertVerifier for CertVerifier {
    fn verify_server_cert(
        &self,
        endpoint_certificate: &rustls::pki_types::CertificateDer<'_>,
        intermediates: &[rustls::pki_types::CertificateDer<'_>],
        host_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        if !intermediates.is_empty() {
            return Err(rustls::Error::InvalidCertificate(
                CertificateError::UnknownIssuer,
            ));
        }
        // if self.trusted_spki.is_empty() {
        //     return Ok(ServerCertVerified::assertion());
        // }
        let end_entity_as_spki = SubjectPublicKeyInfoDer::from(endpoint_certificate.as_ref());

        // TODO: confirm that this end_entity is valid for this server_name.
        match true {
            true => Ok(ServerCertVerified::assertion()),
            false => Err(rustls::Error::InvalidCertificate(
                CertificateError::UnknownIssuer,
            )),
        }
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

    fn requires_raw_public_keys(&self) -> bool {
        // TODO: can we support x.509 certificates as well?
        true
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
