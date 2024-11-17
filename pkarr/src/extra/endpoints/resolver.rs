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
            // TODO: cache the result of this function?
            // TODO: test load balancing
            // TODO: test failover
            // TODO: custom max_chain_length

            let mut step = 0;
            let mut stack: Vec<Endpoint> = Vec::new();

            // Initialize the stack with endpoints from the starting domain.
            if let Ok(tld) = PublicKey::try_from(qname) {
                if let Ok(Some(signed_packet)) = self.resolve(&tld).await {
                    step += 1;
                    stack.extend(Endpoint::parse(&signed_packet, qname, is_svcb));
                }
            }

            while let Some(next) = stack.pop() {
                let current = next.domain();

                // Attempt to resolve the domain as a public key.
                match PublicKey::try_from(current) {
                    Ok(tld) => match self.resolve(&tld).await {
                        Ok(Some(signed_packet)) if step < DEFAULT_MAX_CHAIN_LENGTH => {
                            step += 1;
                            stack.extend(Endpoint::parse(&signed_packet, current, is_svcb));
                        }
                        _ => break, // Stop on resolution failure or chain length exceeded.
                    },
                    // Yield if the domain is not pointing to another Pkarr TLD domain.
                    Err(_) => co.yield_(next).await,
                }
            }
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
