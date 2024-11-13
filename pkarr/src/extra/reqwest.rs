use reqwest::dns::{Addrs, Resolve};

use crate::{Client, PublicKey};

use super::endpoints::EndpointResolver;

use std::net::ToSocketAddrs;

impl Resolve for Client {
    fn resolve(&self, name: reqwest::dns::Name) -> reqwest::dns::Resolving {
        let client = self.clone();
        Box::pin(resolve(client, name))
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "relay"))]
impl Resolve for crate::client::relay::Client {
    fn resolve(&self, name: reqwest::dns::Name) -> reqwest::dns::Resolving {
        let client = self.clone();
        Box::pin(resolve(client, name))
    }
}

async fn resolve(
    client: impl EndpointResolver,
    name: reqwest::dns::Name,
) -> Result<Addrs, Box<dyn std::error::Error + Send + Sync>> {
    let name = name.as_str();

    if PublicKey::try_from(name).is_ok() {
        let endpoint = client.resolve_endpoint(name).await?;

        let addrs: Addrs = Box::new(endpoint.to_socket_addrs().into_iter());

        return Ok(addrs);
    };

    Ok(Box::new(format!("{name}:0").to_socket_addrs().unwrap()))
}

#[cfg(test)]
mod tests {
    use axum::{response::Html, routing::get, Router};
    use rand::Rng;
    use std::net::SocketAddr;

    #[tokio::test]
    async fn http() {
        // Define the route and handler
        let app = Router::new().route("/", get(handler));

        let port: u16 = rand::thread_rng().gen();

        let addr = SocketAddr::from(([127, 0, 0, 1], port));

        axum_server::bind(addr)
            .serve(app.into_make_service())
            .await
            .unwrap();

        async fn handler() -> Html<&'static str> {
            Html("Hello, World!")
        }
    }
}
