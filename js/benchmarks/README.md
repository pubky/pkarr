# Benchmarks

This directory keeps track of the tests, benchmarks and stress testing used to justify some of the choices made in the first MVP.

### compression-vs-encoding

Ran multiple encoding and compression combination on multiple anticipated typical sets of records (documents).

Conclusion: Most documents are expected to be smaller than 500 bytes, so no matter how much you compress, you can't save more than 200 bytes. 

Given that compression doesn't seem to be that benificial, we should choose the simplest scheme that can be implemented in any language, with less dependencies.

- JSON is adds size overhead and an extra dependency.
- Bencode is required anyways, but it also adds size overhead without compression.
- Compression with lz4 is very fast byt it adds overhead instead of saving bytes most of the time.
- Compression with Brotli is great, but adds a complex dependency, and as mentioned the savings aren't worth it.
- CSV is human readable, but is hard to do without a library (too many foot guns).
- A custom simple encoding without compression is probably the best:

Custom encoding:

```
|-----------|-------------|----------|-------------|-------------|
| 2 bytes   | 1 byte      | max 255  | 1 byte      | max 255     |
|-----------|-------------|----------|-------------|-------------|
| Type code | name length | name     | data length | record data |
|-----------|-------------|----------|-------------|-------------|
```

`record data` encoding is UTF-8. We can encode `A` and `AAAA` in a more compact way, but the bytes savings isn't worth the complexity of it.


```bash
 === single record - IP ===
 base size: 46
 ┌--------------------┬----------┬----------┬----------┬------------┐
 | TYPE               | Absolute | Relative | ENCODE   | DECODE     |
 |--------------------|----------|----------|----------|------------|
 | csv - no comp      | +0.00    | +0.00 %  | 0.00 ms  | 0.00 ms    |
 | bencode - no comp  | +9.00    | +0.20 %  | 0.00 ms  | 0.00 ms    |
 | json + lz4         | +31.00   | +0.67 %  | 0.50 ms  | 0.50 ms    |
 | bencode + lz4      | +30.00   | +0.65 %  | 0.50 ms  | 0.50 ms    |
 | csv + lz4          | +21.00   | +0.46 %  | 0.00 ms  | 0.50 ms    |
 | json + brotli      | +14.00   | +0.30 %  | 13.50 ms | 0.00 ms    |
 | bencode + brotli   | +7.00    | +0.15 %  | 8.00 ms  | 0.00 ms    |
 | csv + brotli       | +4.00    | +0.09 %  | 7.50 ms  | 0.00 ms    |
 | custom + brotli    | -15.00   | -0.33 %  | 11.00 ms | 0.00 ms    |
 | custom + no comp   | -19.00   | -0.41 %  | 0.00 ms  | 0.50 ms    |
 └--------------------┴----------┴----------┴----------┴------------┘

 === single record - TXT ===
 base size: 162
 ┌--------------------┬----------┬----------┬----------┬------------┐
 | TYPE               | Absolute | Relative | ENCODE   | DECODE     |
 |--------------------|----------|----------|----------|------------|
 | csv - no comp      | +0.00    | +0.00 %  | 0.00 ms  | 0.00 ms    |
 | bencode - no comp  | +10.00   | +0.06 %  | 0.00 ms  | 0.00 ms    |
 | json + lz4         | +26.00   | +0.16 %  | 0.00 ms  | 0.50 ms    |
 | bencode + lz4      | +26.00   | +0.16 %  | 0.00 ms  | 0.50 ms    |
 | csv + lz4          | +16.00   | +0.10 %  | 0.00 ms  | 0.50 ms    |
 | json + brotli      | -14.00   | -0.09 %  | 9.00 ms  | 0.00 ms    |
 | bencode + brotli   | -16.00   | -0.10 %  | 8.00 ms  | 0.00 ms    |
 | csv + brotli       | -27.00   | -0.17 %  | 8.50 ms  | 0.50 ms    |
 | custom + brotli    | -25.00   | -0.15 %  | 7.50 ms  | 0.50 ms    |
 | custom + no comp   | -1.00    | -0.01 %  | 0.00 ms  | 0.00 ms    |
 └--------------------┴----------┴----------┴----------┴------------┘

 === multiple records - TXT - compact text ===
 base size: 212
 ┌--------------------┬----------┬----------┬----------┬------------┐
 | TYPE               | Absolute | Relative | ENCODE   | DECODE     |
 |--------------------|----------|----------|----------|------------|
 | csv - no comp      | +0.00    | +0.00 %  | 0.00 ms  | 0.00 ms    |
 | bencode - no comp  | +23.00   | +0.11 %  | 0.00 ms  | 0.00 ms    |
 | json + lz4         | +26.00   | +0.12 %  | 0.50 ms  | 0.50 ms    |
 | bencode + lz4      | +34.00   | +0.16 %  | 0.00 ms  | 0.50 ms    |
 | csv + lz4          | +11.00   | +0.05 %  | 0.00 ms  | 0.00 ms    |
 | json + brotli      | -27.00   | -0.13 %  | 9.00 ms  | 0.50 ms    |
 | bencode + brotli   | -22.00   | -0.10 %  | 8.50 ms  | 0.00 ms    |
 | csv + brotli       | -41.00   | -0.19 %  | 8.50 ms  | 0.00 ms    |
 | custom + brotli    | -37.00   | -0.17 %  | 9.50 ms  | 0.00 ms    |
 | custom + no comp   | -5.00    | -0.02 %  | 0.00 ms  | 0.00 ms    |
 └--------------------┴----------┴----------┴----------┴------------┘

 === medium - mix of A, AAAA, CNAME and TXT ===
 base size: 357
 ┌--------------------┬----------┬----------┬----------┬------------┐
 | TYPE               | Absolute | Relative | ENCODE   | DECODE     |
 |--------------------|----------|----------|----------|------------|
 | csv - no comp      | +0.00    | +0.00 %  | 0.00 ms  | 0.00 ms    |
 | bencode - no comp  | +52.00   | +0.15 %  | 0.00 ms  | 0.00 ms    |
 | json + lz4         | +25.00   | +0.07 %  | 0.50 ms  | 0.50 ms    |
 | bencode + lz4      | +43.00   | +0.12 %  | 0.00 ms  | 0.50 ms    |
 | csv + lz4          | -5.00    | -0.01 %  | 0.00 ms  | 0.00 ms    |
 | json + brotli      | -67.00   | -0.19 %  | 9.00 ms  | 0.50 ms    |
 | bencode + brotli   | -66.00   | -0.18 %  | 9.00 ms  | 0.00 ms    |
 | csv + brotli       | -87.00   | -0.24 %  | 9.00 ms  | 0.00 ms    |
 | custom + brotli    | -102.00  | -0.29 %  | 13.50 ms | 0.00 ms    |
 | custom + no comp   | -54.00   | -0.15 %  | 0.00 ms  | 0.00 ms    |
 └--------------------┴----------┴----------┴----------┴------------┘

 === multiple records - A, AAA ===
 base size: 104
 ┌--------------------┬----------┬----------┬----------┬------------┐
 | TYPE               | Absolute | Relative | ENCODE   | DECODE     |
 |--------------------|----------|----------|----------|------------|
 | csv - no comp      | +0.00    | +0.00 %  | 0.00 ms  | 0.00 ms    |
 | bencode - no comp  | +22.00   | +0.21 %  | 0.00 ms  | 0.00 ms    |
 | json + lz4         | +30.00   | +0.29 %  | 0.00 ms  | 0.50 ms    |
 | bencode + lz4      | +35.00   | +0.34 %  | 0.00 ms  | 0.50 ms    |
 | csv + lz4          | +12.00   | +0.12 %  | 0.00 ms  | 0.50 ms    |
 | json + brotli      | -3.00    | -0.03 %  | 17.00 ms | 0.00 ms    |
 | bencode + brotli   | -12.00   | -0.12 %  | 15.50 ms | 0.50 ms    |
 | csv + brotli       | -18.00   | -0.17 %  | 15.00 ms | 0.00 ms    |
 | custom + brotli    | -36.00   | -0.35 %  | 14.00 ms | 0.00 ms    |
 | custom + no comp   | -40.00   | -0.38 %  | 0.00 ms  | 0.00 ms    |
 └--------------------┴----------┴----------┴----------┴------------┘

 === large - text heavy ===
 base size: 459
 ┌--------------------┬----------┬----------┬----------┬------------┐
 | TYPE               | Absolute | Relative | ENCODE   | DECODE     |
 |--------------------|----------|----------|----------|------------|
 | csv - no comp      | +0.00    | +0.00 %  | 0.00 ms  | 0.00 ms    |
 | bencode - no comp  | +48.00   | +0.10 %  | 0.00 ms  | 0.00 ms    |
 | json + lz4         | -3.00    | -0.01 %  | 0.00 ms  | 0.50 ms    |
 | bencode + lz4      | +18.00   | +0.04 %  | 0.00 ms  | 0.50 ms    |
 | csv + lz4          | -28.00   | -0.06 %  | 0.00 ms  | 0.50 ms    |
 | json + brotli      | -162.00  | -0.35 %  | 16.50 ms | 0.00 ms    |
 | bencode + brotli   | -156.00  | -0.34 %  | 15.00 ms | 0.50 ms    |
 | csv + brotli       | -176.00  | -0.38 %  | 15.00 ms | 0.00 ms    |
 | custom + brotli    | -177.00  | -0.39 %  | 16.00 ms | 0.50 ms    |
 | custom + no comp   | -35.00   | -0.08 %  | 0.00 ms  | 0.00 ms    |
 └--------------------┴----------┴----------┴----------┴------------┘
```
