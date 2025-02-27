use ntimestamp::Timestamp;

use crate::{Cache, PublicKey, SignedPacket};

use super::{Client, PublishError};

impl Client {
    /// Returns a blocking (synchronous ) version of [Client].
    pub fn as_blocking(&self) -> ClientBlocking {
        ClientBlocking(self.clone())
    }
}

/// A blocking (synchronous) version of [Client].
pub struct ClientBlocking(Client);

impl ClientBlocking {
    // === Getters ===

    /// Returns a reference to the internal cache.
    pub fn cache(&self) -> Option<&dyn Cache> {
        self.0.cache()
    }

    /// Returns a reference to the internal [mainline::Dht] node.
    ///
    /// Gives you access to methods like [mainline::Dht::info],
    /// [mainline::Dht::bootstrapped], and [mainline::Dht::to_bootstrap]
    /// among the rest of the API.
    #[cfg(dht)]
    pub fn dht(&self) -> Option<mainline::Dht> {
        self.0.dht()
    }

    // === Publish ===

    /// Publishes a [SignedPacket] to the [mainline] Dht and or [Relays](https://pkarr.org/relays).
    ///
    /// # Lost Update Problem
    ///
    /// Mainline DHT and remote relays form a distributed network, and like all distributed networks,
    /// it is vulnerable to [Write–write conflict](https://en.wikipedia.org/wiki/Write-write_conflict).
    ///
    /// ## Read first
    ///
    /// To mitigate the risk of lost updates, you should call the [Self::resolve_most_recent] method
    /// then start authoring the new [SignedPacket] based on the most recent as in the following example:
    ///
    ///```rust
    /// use pkarr::{Client, SignedPacket, Keypair};
    /// // For local testing
    /// use pkarr::mainline::Testnet;
    ///
    /// fn run() -> anyhow::Result<()> {
    ///     let testnet = Testnet::new(3)?;
    ///     let client = Client::builder()
    ///         // Disable the default network settings (builtin relays and mainline bootstrap nodes).
    ///         .no_default_network()
    ///         .bootstrap(&testnet.bootstrap)
    ///         .build()?
    ///         .as_blocking();
    ///
    ///     let keypair = Keypair::random();
    ///
    ///     let (signed_packet, cas) = if let Some(most_recent) = client
    ///         .resolve_most_recent(&keypair.public_key())
    ///     {
    ///
    ///         let mut builder = SignedPacket::builder();
    ///
    ///         // 1. Optionally inherit all or some of the existing records.
    ///         for record in most_recent.all_resource_records() {
    ///             let name = record.name.to_string();
    ///
    ///             if name != "foo" && name != "sercert" {
    ///                 builder = builder.record(record.clone());
    ///             }
    ///         };
    ///
    ///         // 2. Optionally add more new records.
    ///         let signed_packet = builder
    ///             .txt("foo".try_into()?, "bar".try_into()?, 30)
    ///             .a("secret".try_into()?, 42.into(), 30)
    ///             .sign(&keypair)?;
    ///
    ///         (
    ///             signed_packet,
    ///             // 3. Use the most recent [SignedPacket::timestamp] as a `CAS`.
    ///             Some(most_recent.timestamp())
    ///         )
    ///     } else {
    ///         (
    ///             SignedPacket::builder()
    ///                 .txt("foo".try_into()?, "bar".try_into()?, 30)
    ///                 .a("secret".try_into()?, 42.into(), 30)
    ///                 .sign(&keypair)?,
    ///             None
    ///         )
    ///     };
    ///
    ///     client.publish(&signed_packet, cas)?;
    ///
    ///     Ok(())
    /// }
    /// ```
    ///
    /// ## Errors
    ///
    /// This method may return on of these errors:
    ///
    /// 1. [super::QueryError]: when the query fails, and you need to retry or debug the network.
    /// 2. [super::ConcurrencyError]: when an write conflict (or the risk of it) is detedcted.
    ///
    /// If you get a [super::ConcurrencyError]; you should resolver the most recent packet again,
    /// and repeat the steps in the previous example.
    pub fn publish(
        &self,
        signed_packet: &SignedPacket,
        cas: Option<Timestamp>,
    ) -> Result<(), PublishError> {
        futures_lite::future::block_on(self.0.publish(signed_packet, cas))
    }

    // === Resolve ===

    /// Returns a [SignedPacket] from the cache even if it is expired.
    /// If there is no packet in the cache, or if the cached packet is expired,
    /// it will make a DHT query in a background query and caches any more recent packets it receives.
    ///
    /// If you want to get the most recent version of a [SignedPacket],
    /// you should use [Self::resolve_most_recent].
    pub fn resolve(&self, public_key: &PublicKey) -> Option<SignedPacket> {
        futures_lite::future::block_on(self.0.resolve(public_key))
    }

    /// Returns the most recent [SignedPacket] found after querying all
    /// [mainline] Dht nodes and or [Relays](https:://pkarr.org/relays).
    ///
    /// Useful if you want to read the most recent packet before publishing
    /// a new packet.
    ///
    /// This is a best effort, and doesn't guarantee consistency.
    pub fn resolve_most_recent(&self, public_key: &PublicKey) -> Option<SignedPacket> {
        futures_lite::future::block_on(self.0.resolve_most_recent(public_key))
    }
}
