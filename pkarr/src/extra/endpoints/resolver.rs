//! EndpointResolver trait

use futures_lite::{pin, Stream, StreamExt};
use genawaiter::sync::Gen;

use crate::{PublicKey, SignedPacket};

use super::Endpoint;

const DEFAULT_MAX_CHAIN_LENGTH: u8 = 3;

pub trait EndpointsResolver {
    /// Returns an async stream of HTTPS [Endpoint]s
    fn resolve_https_endpoints(&self, qname: &str) -> impl Stream<Item = Endpoint> {
        self.resolve_endpoints(qname, false)
    }

    /// Returns an async stream of either [HTTPS] or [SVCB] [Endpoint]s
    fn resolve_endpoints(&self, qname: &str, is_svcb: bool) -> impl Stream<Item = Endpoint> {
        Gen::new(|co| async move {
            let target = qname;
            // TODO: cache the result of this function?

            let mut step = 0;
            let mut svcb: Option<Endpoint> = None;

            loop {
                let current = svcb
                    .clone()
                    .map_or(target.to_string(), |s| s.target().to_string());
                if let Ok(tld) = PublicKey::try_from(current.clone()) {
                    if let Ok(Some(signed_packet)) = self.resolve(&tld).await {
                        if step >= DEFAULT_MAX_CHAIN_LENGTH {
                            break;
                        };
                        step += 1;

                        // Choose most prior SVCB record
                        svcb = Endpoint::find(&signed_packet, &current, is_svcb);

                        // TODO: support wildcard?
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }

            if let Some(svcb) = svcb {
                if PublicKey::try_from(svcb.target()).is_err() {
                    co.yield_(svcb).await
                }
            }

            // co.yield_(None).await
        })
    }

    /// Helper method that returns the first `HTTPS` [Endpoint] in the Async stream from [EndpointResolver::resolve_endpoints]
    fn resolve_https_endpoint(
        &self,
        qname: &str,
    ) -> impl std::future::Future<Output = Result<Endpoint, FailedToResolveEndpoint>> {
        async move {
            let stream = self.resolve_https_endpoints(qname);

            pin!(stream);

            match stream.next().await {
                Some(endpoint) => Ok(endpoint),
                None => Err(FailedToResolveEndpoint),
            }
        }
    }

    /// A wrapper around the specific Pkarr client's resolve method.
    fn resolve(
        &self,
        public_key: &PublicKey,
    ) -> impl std::future::Future<Output = Result<Option<SignedPacket>, ResolveError>>;
}

#[derive(thiserror::Error, Debug)]
/// Resolve Error from a client
pub enum ResolveError {
    ClientWasShutdown,
    #[cfg(any(target_arch = "wasm32", feature = "relay"))]
    Reqwest(reqwest::Error),
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Resolve endpoint error from the client::resolve {:?}",
            self
        )
    }
}

#[derive(Debug)]
pub struct FailedToResolveEndpoint;

impl std::error::Error for FailedToResolveEndpoint {}

impl std::fmt::Display for FailedToResolveEndpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Could not resolve clear net endpoint for the Pkarr domain"
        )
    }
}
