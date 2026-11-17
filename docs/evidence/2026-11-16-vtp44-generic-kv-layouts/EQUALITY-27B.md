# Qwen3.8-27B: default caches unchanged, packed caches re-measured on a same-build reference (VTP44)

## Summary
With rad0037/38/40/41/42 applied, the 27B default bf16 and fp8 caches give the stock build's output to the KLD tool's
float floor, and the packed caches reproduce VTP43's long-corpus numbers exactly. bf16_i8 still meets the 0.0027 bar
(0.002685) on a reference the stock build recorded. The short-corpus figures VTP43 quoted for the packed caches were
measured against an older build's reference and were too low; the corrected ones are below.

## Outcome
`qwen3.8-27b-w8a8-df2.rad`, TP3 over two 7900 XTX and an R9700, exact wire, no speculation. The stock build
(`radiance:stock:linux-radiance` at origin/main bdb299751, no experiment) recorded each bf16 reference; the
`kv-bf16-packed` build scored every arm against it.

Default caches (identity), against the stock build's reference for the same cache:

| cache | corpus | mean KLD | max | top-1 |
|---|---|---|---|---|
| bf16 | short, 42,761 positions | 6.1e-11 | 2.5e-08 | 100% |
| fp8 | short | 4.7e-11 | 2.4e-08 | 100% |
| bf16 | long, 276,900 positions | 4.8e-11 | - | 100% |

Packed caches against the stock build's bf16 reference:

| cache | short mean / p99.9 / top-1 | VTP43 short | long mean / p99.9 / top-1 | VTP43 long |
|---|---|---|---|---|
| fp8 | 0.002235 / 0.084 / 97.49% | 0.00221 | 0.004947 / 0.591 / 98.23% | 0.00495 |
| bf16_fp8 | 0.001922 / 0.046 / 98.05% | 0.00157 | 0.003228 / 0.318 / 98.62% | 0.00323 |
| bf16_i8 | 0.001411 / 0.043 / 98.12% | 0.00111 | 0.002685 / 0.186 / 98.67% | 0.00269 |

Long corpus, per document p99.9 (bf16_i8 / bf16_fp8 / fp8): doc 00 1.87 / 2.46 / 3.92; doc 02 3.33 / 4.71 / 6.13; the
other three documents 0.02 - 0.04 for all three. The two documents above 1 are the ones VTP43 recorded and the owner
accepted.

## Detail
The old short reference. VTP43's short-corpus packed-cache runs scored against a reference recorded by an older build
(before the production patches that merged since). Today's bf16 cache scores 0.00085 against that reference, so those
figures carried a cross-build offset; VTP43's fp8 short figure was recorded on the reference's own build and was
unaffected. The long corpus was not affected: its figures reproduce to the digit. The order bf16_i8 < bf16_fp8 < fp8 holds
in both corpora. Use the table above, not VTP43's short column.

Capacity and speed are VTP43's; they were not re-measured here. The libr3 gfx1100 device selftest for patch B's rows was
not run (the rows were exercised on gfx1100 by Flash-Next serving on the two 7900 XTX, FLASHNEXT-AB.md).
