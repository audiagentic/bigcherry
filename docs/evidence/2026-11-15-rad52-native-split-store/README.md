# RAD52: libr3 gfx1100 native prefill attention serves KV-split launches

2026-11-15. Layout `qwen27b-tp3-int8-556-long-bf16` (TP3: 7900 XTX, 7900 XTX, R9700; split 5,5,6; one KV head on each
XTX rank). Engine argv from the stored manifest; the RAD52 arm adds only `--radiance-home` pointing at the RAD52 libr3.

## Change

libr3's native gfx1100 prefill attention refused the KV split (`static_assert(!R3N || !(OPT & O_PSPLIT))`), so a split
launch ran the upstream (emulated-fragment) form. An XTX rank holding one KV head launches a 1024-row piece as 16
workgroups on 48 WGPs, so at depth that rank's attention was on the critical path (the quadratic TTFT term, 18.7 s at
100k tokens; see `docs/evidence/2026-11-14-tp3-long-context/`). RAD52 converts the native accumulator to the decode
partial layout before the split store and selects the native form for the split launch when `attn_native` is set.

## Correctness

- `r3_selftest --case attn_prefill_split` on gfx1100, including a new one-KV-head (TP3 rank) geometry for all four cache
  formats and the production pieces (two 1024-row pieces of a 2048-token step at 50k keys; a 2048-row step at 100k
  keys): 2382 ok, 0 fail. Split vs unsplit max difference: 0.00195 (k6_v4, fp8_v4), 1.5e-5 (fp8), 1.2e-4 (bf16).
- Long-corpus KLD (5 documents, 276,900 positions, top-64 + tail reference recorded from stock libr3 on the same argv,
  one sequence, no speculation). The noise floor is stock with `--pipelined-prefill off` against the same reference.

| arm | KLD mean | p99 | p99.9 | top-1 agreement | PPL ratio |
|---|---|---|---|---|---|
| RAD52 | 0.00315 | 0.00569 | 0.209 | 0.9870 | 1.0001 |
| noise floor (stock, serial prefill) | 0.00283 | 0.00555 | 0.183 | 0.9871 | 0.9990 |

  Per document, p99 and top-1 match the noise floor everywhere; the mean is higher on the two 25k-token documents
  (00: 0.0089 vs 0.0076, 02: 0.0190 vs 0.0164), equal on two, lower on the 100k document (04: 0.00101 vs 0.00115).
  Those two documents carry the corpus's rare large divergences (max KLD ~22 in both arms). Read: at the noise floor.
- Stress: 12 fresh starts of the RAD52 engine, each running 4- and 8-stream benches (512-token prompts, 128 out) and
  4 decoding streams beside a 60k-token prompt: 12/12 up and alive, 0 fault lines, every bench ok.

## Performance (ABBA, 2 starts per arm, served argv)

| metric | stock | RAD52 | change |
|---|---|---|---|
| TTFT 100k tokens | 53.75 ± 0.33 s | 43.97 ± 0.03 s | -18.2% |
| TTFT 60k | 27.93 s | 24.11 s | -13.7% |
| TTFT 30k | 12.22 s | 11.17 s | -8.6% |
| TTFT 8k | 2.90 s | 2.90 s | +0.1% |
| decode, 1 stream (chunks/s) | 33.70 | 33.72 | 0% |
| decode, 8 streams (chunks/s) | 88.49 | 89.52 ± 2.00 | +1.2% (within spread) |
| 100k prompt beside 4 decoders: long TTFT | 61.56 s | 51.07 s | -17.0% |
| decode beside it (chunks/s) | 5.00 | 6.00 | +20% |
| longest decode gap | 1540 ms | 1155 ms | -25% |

Nothing changes at 8k tokens or in decode: only split launches change, and the gain grows with depth (-8.6% at 30k,
-18.2% at 100k). A third arm (RAD52 plus `--prefill-span 2` wide steps, one start) measured about the same as RAD52 alone;
the two remove overlapping time, so RAD52 is the one taken.

## Final repeat and per-row flatness (main libr3)

A second ABBA x2 on the served `-long-bf16` argv (A stock, R main's libr3 with RAD52 and RAD53; RAD53 changes nothing
on a bf16 cache), 4 fresh starts per arm, adding 512- and 2048-token prompts:

| metric | stock | main libr3 | change |
|---|---|---|---|
| TTFT 100k tokens | 53.94 s | 43.91 s | -18.6% |
| TTFT 60k / 30k | 27.94 / 12.22 s | 24.08 / 11.18 s | -13.8% / -8.5% |
| TTFT 8k / 2048 / 512 | 2.920 / 0.813 / 0.243 s | 2.891 / 0.820 / 0.243 s | -1.0% / +0.9% / -0.2% |
| decode, 1 / 8 streams (chunks/s, decode only) | 36.49 / 144.5 | 36.48 / 144.0 | -0.1% / -0.4% |
| 100k prompt beside 4 decoders: long TTFT | 61.71 s | 51.35 s | -16.8% |
| decode beside it / longest gap | 4.98 chunks/s / 1537 ms | 5.98 / 1159 ms | +20% / -25% |

Per query row at one KV head (an XTX rank's TP3 geometry: `r3_selftest --perfpaged-kv 1 --perfpaged Q CTX 5`, one XTX,
best of 2, us a row):

| keys | cache | stock 256 / 512 / 1024 / 2048 / 4096 | main 256 / 512 / 1024 / 2048 / 4096 | main spread 256-2048 |
|---|---|---|---|---|
| 50k | bf16 | 11.5 / 11.8 / 11.4 / 11.3 / 7.5 | 6.2 / 6.1 / 5.8 / 5.7 / 7.6 | 7% |
| 50k | fp8 | 18.4 / 18.0 / 17.9 / 19.3 / 13.9 | 5.4 / 5.1 / 5.0 / 4.9 / 6.5 | 11% |
| 50k | fp8_v4 | 10.6 / 11.0 / 10.8 / 10.5 / 6.2 | 5.3 / 4.8 / 4.7 / 4.6 / 6.0 | 14% |
| 50k | k6_v4 | 19.9 / 19.7 / 22.2 / 21.0 / 13.3 | 5.3 / 4.8 / 4.7 / 4.6 / 5.8 | 15% |
| 100k | bf16 | 24.3 / 23.7 / 22.9 / 23.5 / 15.5 | 12.3 / 11.9 / 12.0 / 12.4 / 15.6 | 5% |
| 100k | fp8 | 36.2 / 36.0 / 38.9 / 38.5 / 29.0 | 10.3 / 10.0 / 9.8 / 9.9 / 13.2 | 5% |
| 100k | fp8_v4 | 22.2 / 21.7 / 20.8 / 21.8 / 12.8 | 9.5 / 9.4 / 9.2 / 9.6 / 12.2 | 4% |
| 100k | k6_v4 | 39.0 / 41.6 / 42.2 / 42.3 / 27.5 | 9.6 / 9.3 / 9.2 / 9.6 / 11.9 | 5% |

From 256 to 2048 rows the cost per row is flat within 4-15% on main and 2-4x below stock. At 4096 rows (no split
needed) main costs about 30% more a row than at 2048, and there stock's unsplit path is cheaper than its own split one;
the engine's steps are 2048 rows, so this does not arise in serving.
