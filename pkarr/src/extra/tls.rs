use std::{fmt::Debug, sync::Arc};

use rustls::{
    client::{
        danger::{DangerousClientConfigBuilder, ServerCertVerified, ServerCertVerifier},
        WebPkiServerVerifier,
    },
    crypto::{verify_tls13_signature_with_raw_key, WebPkiSupportedAlgorithms},
    pki_types::SubjectPublicKeyInfoDer,
    SignatureScheme,
};

use crate::{Client, PublicKey};

use crate::extra::endpoints::EndpointsResolver;

#[derive(Debug)]
struct CertVerifier<T: EndpointsResolver + Send + Sync + Debug> {
    pkarr_client: T,
    webpki: Arc<WebPkiServerVerifier>,
}

static SUPPORTED_ALGORITHMS: WebPkiSupportedAlgorithms = WebPkiSupportedAlgorithms {
    all: &[webpki::ring::ED25519],
    mapping: &[(SignatureScheme::ED25519, &[webpki::ring::ED25519])],
};

impl<T: EndpointsResolver + Send + Sync + Debug> ServerCertVerifier for CertVerifier<T> {
    /// Verify Pkarr public keys
    fn verify_server_cert(
        &self,
        endpoint_certificate: &rustls::pki_types::CertificateDer<'_>,
        intermediates: &[rustls::pki_types::CertificateDer<'_>],
        host_name: &rustls::pki_types::ServerName<'_>,
        ocsp_response: &[u8],
        now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        if PublicKey::try_from(host_name.to_str().as_ref()).is_err() {
            return self.webpki.verify_server_cert(
                endpoint_certificate,
                intermediates,
                host_name,
                ocsp_response,
                now,
            );
        }
        let end_entity_as_spki = SubjectPublicKeyInfoDer::from(endpoint_certificate.as_ref());

        // TODO: confirm that this end_entity is valid for this server_name.
        dbg!(host_name, end_entity_as_spki);

        Ok(ServerCertVerified::assertion())
        // match true {
        //     true => Ok(ServerCertVerified::assertion()),
        //     false => Err(rustls::Error::InvalidCertificate(
        //         CertificateError::UnknownIssuer,
        //     )),
        // }
    }

    /// Same as [WebPkiServerVerifier::verify_tls12_signature]
    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        // TODO: fallback to webpki
        // self.webpki.verify_tls12_signature(message, cert, dss)

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
        // TODO: fallback to webpki
        // self.webpki.verify_tls13_signature(message, cert, dss)

        verify_tls13_signature_with_raw_key(
            message,
            &SubjectPublicKeyInfoDer::from(cert.as_ref()),
            dss,
            &SUPPORTED_ALGORITHMS,
        )
    }

    /// Same as [WebPkiServerVerifier::supported_verify_schemes]
    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        self.webpki.supported_verify_schemes()
    }

    fn requires_raw_public_keys(&self) -> bool {
        // TODO: can we change this to false and still work for pkarr domains?
        true
    }
}

impl<T: EndpointsResolver + Send + Sync + Debug> CertVerifier<T> {
    pub(crate) fn new(pkarr_client: T) -> Self {
        let webpki = WebPkiServerVerifier::builder_with_provider(
            rustls::RootCertStore {
                roots: webpki_roots::TLS_SERVER_ROOTS.to_vec(),
            }
            .into(),
            rustls::crypto::ring::default_provider().into(),
        )
        .build()
        .expect("WebPkiServerVerifier build");

        CertVerifier {
            pkarr_client,
            webpki,
        }
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
    fn from(client: Client) -> Self {
        let verifier: CertVerifier<Client> = client.into();

        create_client_config_with_ring()
            .with_custom_certificate_verifier(Arc::new(verifier))
            .with_no_client_auth()
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "relay"))]
impl From<crate::client::relay::Client> for rustls::ClientConfig {
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
