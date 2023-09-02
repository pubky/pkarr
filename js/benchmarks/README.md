# Benchmarks

This directory keeps track of the tests, benchmarks and stress testing used to justify some of the choices made in the first MVP.

### compression-vs-encoding

First attempt: ~Assess if using CBOR is worth it instead of JSON compression using brtoli.~ CBOR wasn't really worth it

Second attempt: Do we need JSON encoding (which doesn't come for free in all languages like Javascript)?

Results:

```
 ┌--------------------┬------------┬----------┬------------┐
 | TYPE               | SIZE RATIO | COMPRESS | DECOMPRESS |
 |--------------------|------------|----------|------------|
 | json + brotli      | 0.48       | 1 ms     | 1 ms       |
 | bencode + lz4      | 0.52       | 0 ms     | 1 ms       |
 | csv + lz4          | 0.49       | 0 ms     | 1 ms       |
 | bencode + brotli   | 0.34       | 26 ms    | 0 ms       |
 | csv + brotli       | 0.33       | 20 ms    | 1 ms       |
 | custom + brotli    | 0.31       | 14 ms    | 4 ms       |
 └--------------------┴------------┴----------┴------------┘
```

Conclusion:

`json + brotli` is very fast, but doesn't compress data enough.

`csv + brotli` seems to offer the best compression, fast decoding, and acceptable encoding time.

`custom + brotli` Where we encode dns type by its code ~dosen't offer much benifit, it needs two bytes, which only saves 1 byte for `TXT` and 2 for `AAAA`, while complecating decoding.~ 

Actually encoding ipv4 and ipv6 for what is mostly `A`, `AAAA`, and `TXT` records, it starts to pay off.

```
 ┌--------------------┬------------┬----------┬------------┐
 | TYPE               | SIZE RATIO | COMPRESS | DECOMPRESS |
 |--------------------|------------|----------|------------|
 | json + brotli      | 0.97       | 1 ms     | 1 ms       |
 | bencode + lz4      | 1.01       | 0 ms     | 1 ms       |
 | csv + lz4          | 0.96       | 0 ms     | 1 ms       |
 | bencode + brotli   | 0.78       | 21 ms    | 0 ms       |
 | csv + brotli       | 0.77       | 10 ms    | 0 ms       |
 | custom + brotli    | 0.60       | 13 ms    | 5 ms       |
 └--------------------┴------------┴----------┴------------┘
 ```
