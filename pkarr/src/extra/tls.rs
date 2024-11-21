use std::{fmt::Debug, sync::Arc};

use futures_lite::{pin, stream::block_on};
use rustls::{
    client::danger::{DangerousClientConfigBuilder, ServerCertVerified, ServerCertVerifier},
    crypto::{verify_tls13_signature_with_raw_key, WebPkiSupportedAlgorithms},
    pki_types::SubjectPublicKeyInfoDer,
    CertificateError, SignatureScheme,
};

use crate::Client;

use crate::extra::endpoints::EndpointsResolver;

#[derive(Debug)]
pub struct CertVerifier<T: EndpointsResolver + Send + Sync + Debug + Clone>(T);

static SUPPORTED_ALGORITHMS: WebPkiSupportedAlgorithms = WebPkiSupportedAlgorithms {
    all: &[webpki::ring::ED25519],
    mapping: &[(SignatureScheme::ED25519, &[webpki::ring::ED25519])],
};

impl<T: EndpointsResolver + Send + Sync + Debug + Clone> ServerCertVerifier for CertVerifier<T> {
    /// Verify Pkarr public keys
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

        let end_entity_as_spki = SubjectPublicKeyInfoDer::from(endpoint_certificate.as_ref());
        let expected_spki = end_entity_as_spki.as_ref();

        let qname = host_name.to_str();

        // Resolve HTTPS endpoints and hope that the cached SignedPackets didn't chance
        // since the last time we resolved endpoints to establish the connection in the
        // first place.
        let stream = self.0.resolve_https_endpoints(&qname);
        pin!(stream);
        for endpoint in block_on(stream) {
            if endpoint.public_key().to_public_key_der().as_bytes() == expected_spki {
                return Ok(ServerCertVerified::assertion());
            }
        }

        // Repeat for SVCB endpoints
        let stream = self.0.resolve_svcb_endpoints(&qname);
        pin!(stream);
        for endpoint in block_on(stream) {
            if endpoint.public_key().to_public_key_der().as_bytes() == expected_spki {
                return Ok(ServerCertVerified::assertion());
            }
        }

        Err(rustls::Error::InvalidCertificate(
            CertificateError::UnknownIssuer,
        ))
    }

    /// Same as [WebPkiServerVerifier::verify_tls12_signature]
    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        verify_tls13_signature_with_raw_key(
            message,
            &SubjectPublicKeyInfoDer::from(cert.as_ref()),
            dss,
            &SUPPORTED_ALGORITHMS,
        )
    }

    /// Same as [WebPkiServerVerifier::verify_tls13_signature]
    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        verify_tls13_signature_with_raw_key(
            message,
            &SubjectPublicKeyInfoDer::from(cert.as_ref()),
            dss,
            &SUPPORTED_ALGORITHMS,
        )
    }

    /// Same as [WebPkiServerVerifier::supported_verify_schemes]
    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        vec![SignatureScheme::ED25519]
    }

    fn requires_raw_public_keys(&self) -> bool {
        true
    }
}

impl<T: EndpointsResolver + Send + Sync + Debug + Clone> CertVerifier<T> {
    pub(crate) fn new(pkarr_client: T) -> Self {
        CertVerifier(pkarr_client)
    }
}

impl From<Client> for CertVerifier<Client> {
    fn from(pkarr_client: Client) -> Self {
        CertVerifier::new(pkarr_client)
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "relay"))]
impl From<crate::client::relay::Client> for CertVerifier<crate::client::relay::Client> {
    fn from(pkarr_client: crate::client::relay::Client) -> Self {
        CertVerifier::new(pkarr_client)
    }
}

impl From<Client> for rustls::ClientConfig {
    /// Creates a [rustls::ClientConfig] that uses [rustls::crypto::ring::default_provider()]
    /// and no client auth and follows the [tls for pkarr domains](https://pkarr.org/tls) spec.
    ///
    /// If you want more control, create a [CertVerifier] from this [Client] to use as a [custom certificate verifier][DangerousClientConfigBuilder::with_custom_certificate_verifier].
    fn from(client: Client) -> Self {
        let verifier: CertVerifier<Client> = client.into();

        create_client_config_with_ring()
            .with_custom_certificate_verifier(Arc::new(verifier))
            .with_no_client_auth()
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "relay"))]
impl From<crate::client::relay::Client> for rustls::ClientConfig {
    /// Creates a [rustls::ClientConfig] that uses [rustls::crypto::ring::default_provider()]
    /// and no client auth and follows the [tls for pkarr domains](https://pkarr.org/tls) spec.
    ///
    /// If you want more control, create a [CertVerifier] from this [Client] to use as a [custom certificate verifier][DangerousClientConfigBuilder::with_custom_certificate_verifier].
    fn from(client: crate::client::relay::Client) -> Self {
        let verifier: CertVerifier<crate::client::relay::Client> = client.into();

        create_client_config_with_ring()
            .with_custom_certificate_verifier(Arc::new(verifier))
            .with_no_client_auth()
    }
}

fn create_client_config_with_ring() -> DangerousClientConfigBuilder {
    rustls::ClientConfig::builder_with_provider(rustls::crypto::ring::default_provider().into())
        .with_safe_default_protocol_versions()
        .expect("version supported by ring")
        .dangerous()
}
