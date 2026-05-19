//! Publish packets to the DHT, save their public keys, reload the client, then
//! measure cold DHT resolve latency.
//!
//! run this example from the project root:
//!     $ cargo run -p pkarr --example benchmark

use anyhow::Context;
use clap::Parser;
use pkarr::{Client, Keypair, PublicKey, SignedPacket};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Number of packets to publish and resolve.
    #[arg(long, default_value_t = 100)]
    count: usize,

    /// File used to save or load public keys, one key per line.
    #[arg(long, default_value = "pkarr-benchmark-public-keys.txt")]
    keys_file: PathBuf,

    /// Skip publishing and resolve the public keys from --keys-file.
    #[arg(long)]
    resolve_only: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    if !cli.resolve_only && cli.count == 0 {
        anyhow::bail!("--count must be greater than zero");
    }

    let targets = if cli.resolve_only {
        let public_keys = load_public_keys(&cli.keys_file)?;
        println!(
            "Loaded {} public keys from {}",
            public_keys.len(),
            cli.keys_file.display()
        );
        targets_from_public_keys(public_keys)
    } else {
        let client = dht_client()?;
        let packets = benchmark_packets(cli.count)?;

        println!("Publishing {} packets to the DHT ...", packets.len());
        let publish_elapsed = publish_packets(&client, &packets).await?;
        println!("Published in {}", format_duration(publish_elapsed));

        save_public_keys(&cli.keys_file, &packets)?;
        println!("Saved public keys to {}", cli.keys_file.display());

        drop(client);
        targets_from_packets(&packets)
    };

    println!("Reloading DHT client before resolving ...");
    let client = dht_client()?;
    println!("Resolving {} packets from the DHT ...", targets.len());
    let resolve_started = Instant::now();
    let samples = resolve_packets(&client, &targets).await;
    print_report(resolve_started.elapsed(), &samples);

    Ok(())
}

fn dht_client() -> anyhow::Result<Client> {
    Client::builder()
        .no_relays()
        .build()
        .context("failed to build DHT-only client")
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

fn save_public_keys(path: &Path, packets: &[BenchmarkPacket]) -> anyhow::Result<()> {
    let keys = packets
        .iter()
        .map(|packet| packet.public_key.to_string())
        .collect::<Vec<_>>()
        .join("\n");

    fs::write(path, format!("{keys}\n"))
        .with_context(|| format!("failed to write public keys to {}", path.display()))
}

fn load_public_keys(path: &Path) -> anyhow::Result<Vec<PublicKey>> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("failed to read public keys from {}", path.display()))?;
    let public_keys = parse_public_keys(path, &content)?;

    if public_keys.is_empty() {
        anyhow::bail!("{} does not contain any public keys", path.display());
    }

    Ok(public_keys)
}

fn parse_public_keys(path: &Path, content: &str) -> anyhow::Result<Vec<PublicKey>> {
    content
        .lines()
        .enumerate()
        .filter_map(|(index, line)| parse_public_key_line(path, index, line).transpose())
        .collect()
}

fn parse_public_key_line(
    path: &Path,
    index: usize,
    line: &str,
) -> anyhow::Result<Option<PublicKey>> {
    let line = line.trim();
    if line.is_empty() {
        return Ok(None);
    }

    let public_key = line.try_into().with_context(|| {
        format!(
            "invalid public key in {} on line {}",
            path.display(),
            index + 1
        )
    })?;

    Ok(Some(public_key))
}

fn targets_from_packets(packets: &[BenchmarkPacket]) -> Vec<ResolveTarget> {
    packets
        .iter()
        .map(|packet| ResolveTarget {
            public_key: packet.public_key.clone(),
            expected_packet: Some(packet.signed_packet.clone()),
        })
        .collect()
}

fn targets_from_public_keys(public_keys: Vec<PublicKey>) -> Vec<ResolveTarget> {
    public_keys
        .into_iter()
        .map(|public_key| ResolveTarget {
            public_key,
            expected_packet: None,
        })
        .collect()
}

async fn resolve_packets(client: &Client, targets: &[ResolveTarget]) -> Vec<ResolveSample> {
    let mut samples = Vec::with_capacity(targets.len());

    for target in targets {
        samples.push(resolve_packet(client, target).await);
    }

    samples
}

async fn resolve_packet(client: &Client, target: &ResolveTarget) -> ResolveSample {
    let started = Instant::now();
    let packet = client.resolve(&target.public_key).await;
    let latency = started.elapsed();
    let resolved_key = packet
        .as_ref()
        .is_some_and(|packet| packet.public_key() == target.public_key);
    let verified_packet = target.expected_packet.as_ref().map(|expected| {
        packet
            .as_ref()
            .is_some_and(|packet| packet.as_bytes() == expected.as_bytes())
    });

    ResolveSample {
        latency,
        resolved_key,
        verified_packet,
    }
}

fn print_report(total_elapsed: Duration, samples: &[ResolveSample]) {
    let successful_latencies: Vec<Duration> = samples
        .iter()
        .filter(|sample| sample.succeeded())
        .map(|sample| sample.latency)
        .collect();
    let resolved = samples.iter().filter(|sample| sample.resolved_key).count();
    let failures = samples.len() - successful_latencies.len();

    println!("\nResolve results");
    println!("  total:    {}", format_duration(total_elapsed));
    println!("  resolved: {resolved}/{}", samples.len());
    print_verified_count(samples);
    println!("  failures: {failures}");

    if let Some(stats) = LatencyStats::from_latencies(&successful_latencies) {
        println!("\nLatency per successful packet");
        println!("  average: {}", format_millis(stats.average_millis));
        println!("  min:     {}", format_duration(stats.min));
        println!("  p50:     {}", format_duration(stats.p50));
        println!("  p90:     {}", format_duration(stats.p90));
        println!("  p95:     {}", format_duration(stats.p95));
        println!("  p99:     {}", format_duration(stats.p99));
        println!("  max:     {}", format_duration(stats.max));
    }
}

fn print_verified_count(samples: &[ResolveSample]) {
    let verified = samples
        .iter()
        .filter_map(|sample| sample.verified_packet)
        .filter(|verified| *verified)
        .count();
    let verifiable = samples
        .iter()
        .filter(|sample| sample.verified_packet.is_some())
        .count();

    if verifiable > 0 {
        println!("  verified: {verified}/{verifiable}");
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
struct ResolveTarget {
    public_key: PublicKey,
    expected_packet: Option<SignedPacket>,
}

#[derive(Debug)]
struct ResolveSample {
    latency: Duration,
    resolved_key: bool,
    verified_packet: Option<bool>,
}

impl ResolveSample {
    fn succeeded(&self) -> bool {
        self.verified_packet.unwrap_or(self.resolved_key)
    }
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
