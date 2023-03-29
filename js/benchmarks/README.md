# Benchmarks

This directory keeps track of the tests, benchmarks and stress testing used to justify some of the choices made in the first MVP.

### compression-vs-encoding

Assess if using CBOR is worth it instead of JSON compression using brtoli.

```

| Operation     | Size | Ratio | Time     |
|---------------|------|-------|----------|
| JSON          | 3554 | 1     |          |
| CBOR          | 3026 | 0.85  |   3.904ms|
| JSON + lz4    | 1490 | 0.42  |   0.668ms|
| CBOR + lz4    | 1534 | 0.43  |          |
| JSON + brotli | 986  | 0.28  | 165.843ms|
| CBOR + brotli | 1044 | 0.29  |   6.5ms  |
```

JSON + Brotli is slower than CBOR + brotli, but is it enough to justify adding it as a dependency?
