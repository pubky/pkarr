use std::{fmt::Debug, sync::Arc};

use ed25519_dalek::pkcs8::{Document, EncodePrivateKey, EncodePublicKey};
use futures_lite::{pin, stream::block_on};
use rustls::{
    client::danger::{DangerousClientConfigBuilder, ServerCertVerified, ServerCertVerifier},
    crypto::ring::sign::any_eddsa_type,
    crypto::{verify_tls13_signature_with_raw_key, WebPkiSupportedAlgorithms},
    pki_types::CertificateDer,
    pki_types::SubjectPublicKeyInfoDer,
    server::AlwaysResolvesServerRawPublicKeys,
    sign::CertifiedKey,
    CertificateError, ServerConfig, SignatureScheme,
};
use tracing::{instrument, Level};

use crate::{Client, Keypair, PublicKey};

#[derive(Debug)]
pub struct CertVerifier(Client);

static SUPPORTED_ALGORITHMS: WebPkiSupportedAlgorithms = WebPkiSupportedAlgorithms {
    all: &[webpki::ring::ED25519],
    mapping: &[(SignatureScheme::ED25519, &[webpki::ring::ED25519])],
};

impl ServerCertVerifier for CertVerifier {
    #[instrument(ret(level = Level::TRACE), err(level = Level::TRACE))]
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

    #[instrument(ret(level = Level::DEBUG), err(level = Level::DEBUG))]
    /// Verify a message signature using a raw public key and the first TLS 1.3 compatible
    /// supported scheme.
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

    #[instrument(ret(level = Level::DEBUG), err(level = Level::DEBUG))]
    /// Verify a message signature using a raw public key and the first TLS 1.3 compatible
    /// supported scheme.
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

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        vec![SignatureScheme::ED25519]
    }

    fn requires_raw_public_keys(&self) -> bool {
        true
    }
}

impl CertVerifier {
    pub(crate) fn new(pkarr_client: Client) -> Self {
        CertVerifier(pkarr_client)
    }
}

impl From<Client> for CertVerifier {
    fn from(pkarr_client: Client) -> Self {
        CertVerifier::new(pkarr_client)
    }
}

impl From<Client> for rustls::ClientConfig {
    /// Creates a [rustls::ClientConfig] that uses [rustls::crypto::ring::default_provider()]
    /// and no client auth and follows the [tls for pkarr domains](https://pkarr.org/tls) spec.
    ///
    /// If you want more control, create a [CertVerifier] from this [Client] to use as a [custom certificate verifier][DangerousClientConfigBuilder::with_custom_certificate_verifier].
    fn from(client: Client) -> Self {
        let verifier: CertVerifier = client.into();

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

impl Keypair {
    /// Return a RawPublicKey certified key according to [RFC 7250](https://tools.ietf.org/html/rfc7250)
    /// useful to use with [rustls::ConfigBuilder::with_cert_resolver] and [rustls::server::AlwaysResolvesServerRawPublicKeys]
    pub fn to_rpk_certified_key(&self) -> CertifiedKey {
        let client_private_key = any_eddsa_type(
            &self
                .0
                .to_pkcs8_der()
                .expect("Keypair::to_rpk_certificate: convert secret key to pkcs8 der")
                .as_bytes()
                .into(),
        )
        .expect("Keypair::to_rpk_certificate: convert KeyPair to rustls SigningKey");

        let client_public_key = client_private_key
            .public_key()
            .expect("Keypair::to_rpk_certificate: load SPKI");
        let client_public_key_as_cert = CertificateDer::from(client_public_key.to_vec());

        CertifiedKey::new(vec![client_public_key_as_cert], client_private_key)
    }

    #[cfg(all(not(target_family = "wasm"), feature = "tls"))]
    /// Create a [rustls::ServerConfig] using this keypair as a RawPublicKey certificate according to [RFC 7250](https://tools.ietf.org/html/rfc7250)
    pub fn to_rpk_rustls_server_config(&self) -> ServerConfig {
        let cert_resolver =
            AlwaysResolvesServerRawPublicKeys::new(self.to_rpk_certified_key().into());

        ServerConfig::builder_with_provider(rustls::crypto::ring::default_provider().into())
            .with_safe_default_protocol_versions()
            .expect("version supported by ring")
            .with_no_client_auth()
            .with_cert_resolver(std::sync::Arc::new(cert_resolver))
    }
}

impl From<Keypair> for ServerConfig {
    /// calls [Keypair::to_rpk_rustls_server_config]
    fn from(keypair: Keypair) -> Self {
        keypair.to_rpk_rustls_server_config()
    }
}

impl From<&Keypair> for ServerConfig {
    /// calls [Keypair::to_rpk_rustls_server_config]
    fn from(keypair: &Keypair) -> Self {
        keypair.to_rpk_rustls_server_config()
    }
}

impl PublicKey {
    pub fn to_public_key_der(&self) -> Document {
        self.0.to_public_key_der().expect("to_public_key_der")
    }
}
