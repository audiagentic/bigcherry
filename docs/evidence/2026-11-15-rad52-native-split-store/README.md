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
