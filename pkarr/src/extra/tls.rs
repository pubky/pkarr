use std::{fmt::Debug, sync::Arc};

use rustls::{
    client::{
        danger::{ServerCertVerified, ServerCertVerifier},
        ServerCertVerifierBuilder, WebPkiServerVerifier,
    },
    pki_types::SubjectPublicKeyInfoDer,
    CertificateError, SignatureScheme,
};

use crate::{Client, PublicKey};

use crate::extra::endpoints::EndpointsResolver;

#[derive(Debug)]
struct CertVerifier<T: EndpointsResolver + Send + Sync + Debug> {
    pkarr_client: T,
    webpki: Arc<WebPkiServerVerifier>,
}

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
        if let Err(_) = PublicKey::try_from(host_name.to_str().as_ref()) {
            return self.webpki.verify_server_cert(
                endpoint_certificate,
                intermediates,
                host_name,
                ocsp_response,
                now,
            );
        }

        if !intermediates.is_empty() {
            return Err(rustls::Error::InvalidCertificate(
                CertificateError::UnknownIssuer,
            ));
        }
        let end_entity_as_spki = SubjectPublicKeyInfoDer::from(endpoint_certificate.as_ref());

        dbg!(host_name, end_entity_as_spki);

        // TODO: confirm that this end_entity is valid for this server_name.
        match true {
            true => Ok(ServerCertVerified::assertion()),
            false => Err(rustls::Error::InvalidCertificate(
                CertificateError::UnknownIssuer,
            )),
        }
    }

    /// Same as [WebPkiServerVerifier::verify_tls12_signature]
    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        self.webpki.verify_tls12_signature(message, cert, dss)
    }

    /// Same as [WebPkiServerVerifier::verify_tls13_signature]
    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        self.webpki.verify_tls13_signature(message, cert, dss)
    }

    /// Same as [WebPkiServerVerifier::supported_verify_schemes]
    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        self.webpki.supported_verify_schemes()
    }
}

impl<T: EndpointsResolver + Send + Sync + Debug> CertVerifier<T> {
    pub(crate) fn new(pkarr_client: T) -> Self {
        let mut root_cert_store = rustls::RootCertStore::empty();
        root_cert_store.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
        let webpki = WebPkiServerVerifier::builder(root_cert_store.into())
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

        rustls::ClientConfig::builder()
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(verifier))
            .with_no_client_auth()
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "relay"))]
impl From<crate::client::relay::Client> for rustls::ClientConfig {
    fn from(client: crate::client::relay::Client) -> Self {
        let verifier: CertVerifier<crate::client::relay::Client> = client.into();

        rustls::ClientConfig::builder()
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(verifier))
            .with_no_client_auth()
    }
}
