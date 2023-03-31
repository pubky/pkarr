# Benchmarks

This directory keeps track of the tests, benchmarks and stress testing used to justify some of the choices made in the first MVP.

### compression-vs-encoding

Assess if using CBOR is worth it instead of JSON compression using brtoli.

```

| Operation     | Size | Ratio | Time      |
|---------------|------|-------|-----------|
| JSON          | 3448 | 1     |           |
| CBOR          | 2954 | 0.86  |   3.02  ms|
| JSON + lz4    | 1481 | 0.43  |   0.697 ms|
| CBOR + lz4    | 1518 | 0.44  |           |
| JSON + brotli | 994  | 0.29  |  27.657 ms|
| CBOR + brotli | 1044 | 0.29  |  16.075 ms|
| BenC + brotli | 1021 | 0.30  |  15.199 ms|
```

JSON + Bitroli (wasm) seem to be the best combination.
