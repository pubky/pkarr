//! Publish packets to the DHT, clear the local Pkarr cache, then measure cold
//! DHT resolve latency.
//!
//! run this example from the project root:
//!     $ cargo run -p pkarr --example benchmark

use anyhow::Context;
use clap::Parser;
use pkarr::{Cache, CacheKey, Client, Keypair, PublicKey, SignedPacket};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Number of packets to publish and resolve.
    #[arg(long, default_value_t = 100)]
    count: usize,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    if cli.count == 0 {
        anyhow::bail!("--count must be greater than zero");
    }

    let cache = Arc::new(ClearableCache::new(cli.count));
    let client = Client::builder()
        .no_relays()
        .cache(cache.clone())
        .build()
        .context("failed to build DHT-only client")?;
    let packets = benchmark_packets(cli.count)?;

    println!("Publishing {} packets to the DHT ...", packets.len());
    let publish_elapsed = publish_packets(&client, &packets).await?;
    println!("Published in {}", format_duration(publish_elapsed));

    println!("Local cache entries before clear: {}", cache.len());
    cache.clear();
    println!("Local cache entries after clear: {}", cache.len());

    println!("Resolving {} packets from the DHT ...", packets.len());
    let resolve_started = Instant::now();
    let samples = resolve_packets(&client, &packets).await;
    print_report(resolve_started.elapsed(), &samples);

    Ok(())
}

fn benchmark_packets(count: usize) -> anyhow::Result<Vec<BenchmarkPacket>> {
    (0..count).map(benchmark_packet).collect()
}

fn benchmark_packet(index: usize) -> anyhow::Result<BenchmarkPacket> {
    let keypair = Keypair::random();
    let text = format!("pkarr benchmark packet {index}");
    let signed_packet = SignedPacket::builder()
        .txt(
            "_pkarr-benchmark".try_into()?,
            text.as_str().try_into()?,
            30,
        )
        .sign(&keypair)?;

    Ok(BenchmarkPacket {
        public_key: keypair.public_key(),
        signed_packet,
    })
}

async fn publish_packets(client: &Client, packets: &[BenchmarkPacket]) -> anyhow::Result<Duration> {
    let started = Instant::now();

    for (index, packet) in packets.iter().enumerate() {
        client
            .publish(&packet.signed_packet, None)
            .await
            .with_context(|| format!("failed to publish packet {}", index + 1))?;
    }

    Ok(started.elapsed())
}

async fn resolve_packets(client: &Client, packets: &[BenchmarkPacket]) -> Vec<ResolveSample> {
    let mut samples = Vec::with_capacity(packets.len());

    for packet in packets {
        samples.push(resolve_packet(client, packet).await);
    }

    samples
}

async fn resolve_packet(client: &Client, packet: &BenchmarkPacket) -> ResolveSample {
    let started = Instant::now();
    let resolved = client.resolve(&packet.public_key).await;
    let latency = started.elapsed();
    let found_expected_packet = resolved
        .as_ref()
        .is_some_and(|resolved| resolved.as_bytes() == packet.signed_packet.as_bytes());

    ResolveSample {
        latency,
        found_expected_packet,
    }
}

fn print_report(total_elapsed: Duration, samples: &[ResolveSample]) {
    let successful_latencies: Vec<Duration> = samples
        .iter()
        .filter(|sample| sample.found_expected_packet)
        .map(|sample| sample.latency)
        .collect();
    let failures = samples.len() - successful_latencies.len();

    println!("\nResolve results");
    println!("  total:    {}", format_duration(total_elapsed));
    println!(
        "  resolved: {}/{}",
        successful_latencies.len(),
        samples.len()
    );
    println!("  failures: {failures}");

    if let Some(stats) = LatencyStats::from_latencies(&successful_latencies) {
        println!("\nLatency per resolved packet");
        println!("  average: {}", format_millis(stats.average_millis));
        println!("  min:     {}", format_duration(stats.min));
        println!("  p50:     {}", format_duration(stats.p50));
        println!("  p90:     {}", format_duration(stats.p90));
        println!("  p95:     {}", format_duration(stats.p95));
        println!("  p99:     {}", format_duration(stats.p99));
        println!("  max:     {}", format_duration(stats.max));
    }
}

fn format_duration(duration: Duration) -> String {
    format_millis(duration.as_secs_f64() * 1000.0)
}

fn format_millis(millis: f64) -> String {
    format!("{millis:.2} ms")
}

#[derive(Debug)]
struct BenchmarkPacket {
    public_key: PublicKey,
    signed_packet: SignedPacket,
}

#[derive(Debug)]
struct ResolveSample {
    latency: Duration,
    found_expected_packet: bool,
}

#[derive(Debug)]
struct LatencyStats {
    average_millis: f64,
    min: Duration,
    p50: Duration,
    p90: Duration,
    p95: Duration,
    p99: Duration,
    max: Duration,
}

impl LatencyStats {
    fn from_latencies(latencies: &[Duration]) -> Option<Self> {
        if latencies.is_empty() {
            return None;
        }

        let mut sorted = latencies.to_vec();
        sorted.sort();

        Some(Self {
            average_millis: average_millis(&sorted),
            min: sorted[0],
            p50: percentile(&sorted, 50),
            p90: percentile(&sorted, 90),
            p95: percentile(&sorted, 95),
            p99: percentile(&sorted, 99),
            max: sorted[sorted.len() - 1],
        })
    }
}

fn average_millis(latencies: &[Duration]) -> f64 {
    let total_millis: f64 = latencies
        .iter()
        .map(|duration| duration.as_secs_f64() * 1000.0)
        .sum();

    total_millis / latencies.len() as f64
}

fn percentile(sorted: &[Duration], percentile: usize) -> Duration {
    let rank = (sorted.len() * percentile).div_ceil(100);
    sorted[rank.saturating_sub(1)]
}

#[derive(Clone, Debug)]
struct ClearableCache {
    capacity: usize,
    state: Arc<Mutex<CacheState>>,
}

impl ClearableCache {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            state: Arc::new(Mutex::new(CacheState::default())),
        }
    }

    fn clear(&self) {
        let mut state = self.state.lock().expect("ClearableCache mutex");
        state.clear();
    }
}

impl Cache for ClearableCache {
    fn capacity(&self) -> usize {
        self.capacity
    }

    fn len(&self) -> usize {
        self.state
            .lock()
            .expect("ClearableCache mutex")
            .packets
            .len()
    }

    fn put(&self, key: &CacheKey, signed_packet: &SignedPacket) {
        let mut state = self.state.lock().expect("ClearableCache mutex");
        state.put(key, signed_packet, self.capacity);
    }

    fn get(&self, key: &CacheKey) -> Option<SignedPacket> {
        let mut state = self.state.lock().expect("ClearableCache mutex");
        state.get(key)
    }

    fn get_read_only(&self, key: &CacheKey) -> Option<SignedPacket> {
        self.state
            .lock()
            .expect("ClearableCache mutex")
            .packets
            .get(key)
            .cloned()
    }
}

#[derive(Debug, Default)]
struct CacheState {
    packets: HashMap<CacheKey, SignedPacket>,
    order: VecDeque<CacheKey>,
}

impl CacheState {
    fn clear(&mut self) {
        self.packets.clear();
        self.order.clear();
    }

    fn put(&mut self, key: &CacheKey, signed_packet: &SignedPacket, capacity: usize) {
        match self.packets.get_mut(key) {
            Some(existing) if existing.as_bytes() == signed_packet.as_bytes() => {
                existing.set_last_seen(signed_packet.last_seen());
            }
            _ => {
                self.packets.insert(*key, signed_packet.clone());
            }
        }

        self.touch(key);
        self.evict_to_capacity(capacity);
    }

    fn get(&mut self, key: &CacheKey) -> Option<SignedPacket> {
        let packet = self.packets.get(key).cloned();

        if packet.is_some() {
            self.touch(key);
        }

        packet
    }

    fn touch(&mut self, key: &CacheKey) {
        self.order.retain(|existing| existing != key);
        self.order.push_back(*key);
    }

    fn evict_to_capacity(&mut self, capacity: usize) {
        while self.packets.len() > capacity {
            if let Some(key) = self.order.pop_front() {
                self.packets.remove(&key);
            } else {
                return;
            }
        }
    }
}
