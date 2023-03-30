# Benchmarks

This directory keeps track of the tests, benchmarks and stress testing used to justify some of the choices made in the first MVP.

### compression-vs-encoding

Assess if using CBOR is worth it instead of JSON compression using brtoli.

```

| Operation     | Size | Ratio | Time      |
|---------------|------|-------|-----------|
| JSON          | 3448 | 1     |           |
| CBOR          | 2954 | 0.86  |   2.481 ms|
| JSON + lz4    | 1481 | 0.43  |   0.422 ms|
| CBOR + lz4    | 1518 | 0.44  |           |
| JSON + brotli | 991  | 0.29  | 193.036 ms|
| CBOR + brotli | 1044 | 0.29  |   6.160 ms|
| BenC + brotli | 1021 | 0.30  |   6.546 ms|
```

Bencode + Brotli is a bit larger than JSON + Brotli, but 10x faster, and not adding any dependencies that we don't have with BEP44, already.
