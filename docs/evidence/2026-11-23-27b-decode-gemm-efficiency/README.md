# 27B TP3 decode GEMM efficiency at per-rank shapes, and a comparison against a third-party AOT library

Collected 2026-11-23 on release bc-radiance-2020.8.0 (libr3 for gfx1100, libr4d for gfx1201). Attaches to VTP37
(int8 TP3 decode, step 5 "decode GEMV") and reads against VTP36's profile.

**Status:** measurement only. No kernel, patch, layout or default was changed. Every number is a single selftest row
on one card (cold weights), not a served-model measurement. The share of the decode step is an estimate by
subtraction, not a trace.

## Method

- Cards: one RX 7900 XTX (gfx1100) for the libr3 rows, the R9700 (gfx1201) for the libr4d and
  third-party rows. The `gpu` lease was held for every run; one card visible at a time.
- Rows: `r3_selftest <plugin> --perfw8 N K M 30 --cold 256` (the 8-bit trunk rows, `gemm_iq8r_nt_m16` at M 4;
  M 1 and M 4 time the same) and `--perfw4` (the asymmetric W4 x F16 GEMM). `--cold 256` reads 256 MiB through the
  caches before each timed launch, so weights arrive cold as in decode.
- Shapes: Qwen3.8-27B (64 layers, 48 gated delta net and 16 full attention, hidden 5120, FFN 17408, head size 256,
  24 query and 4 KV heads, vocabulary 248320) at rank 0 of `qwen27b-tp3` (kv 1,1,2; ffn 6656,6656,4096; lk 6,6,4;
  lv 18,18,12).

## 1. W4 GEMV against a third-party AOT-compiled library (gfx1201)

N 248320, K 5120, M 1, cold. A model-agnostic W4A16 (group 128) matmul library built by a third party was driven
through its own benchmark entry point with random data; it takes the same shape.

| | time | bytes read | bandwidth |
|---|---|---|---|
| third-party W4A16 matmul | 1037 us | 656 MB (weights + scales) | about 631 GB/s |
| libr4d asymmetric W4 x F16, best of the `--perfw4` sweep | 1053 us | 636 MB | about 600 GB/s |

Parity within 1.5%, with the two reading slightly different byte counts. Both sit near the card's memory bandwidth
limit. The same sweep on the 7900 XTX (libr3) reaches 925 us (about 690 GB/s) at its best entry; the entry the
sweep prints for `wv=1 sk=2` takes 1237 us. Which entry the engine's row rule picks was not traced. The third-party
library is fixed at M 1 and fails at larger M, so no larger-M comparison exists.

## 2. IQ8R decode row at the 27B TP3 rank-0 shapes (7900 XTX)

`gemm_iq8r_nt_m16`, M 4, cold, three repeats each (microseconds, then the median as bandwidth and as a share of the
card's 960 GB/s):

| linear | N x K | weight MB | us (3 repeats) | GB/s | share of 960 |
|---|---|---|---|---|---|
| MLP gate/up | 13312 x 5120 | 68.2 | 90.8 / 90.8 / 90.8 | 751 | 78% |
| MLP down | 5120 x 6656 | 34.1 | 55.3 / 55.1 / 55.3 | 616 | 64% |
| delta net in (qkv, z) | 6144 x 5120 | 31.5 | 48.4 / 49.5 / 49.4 | 637 | 66% |
| delta net out | 5120 x 2304 | 11.8 | 27.5 / 27.4 / 27.8 | 429 | 45% |
| attention q and gate | 3072 x 5120 | 15.7 | 27.7 / 27.9 / 27.9 | 564 | 59% |
| attention k or v | 256 x 5120 | 1.3 | 16.0 / 16.0 / 16.0 | 82 | 9% |
| attention out | 5120 x 1536 | 7.9 | 21.4 / 21.4 / 21.5 | 367 | 38% |

Run to run the spread is about 1%. VTP37's earlier isolated decode GEMV figure (690-715 GB/s on the XTX, 72-94% of
nominal) was taken at the model-sized fixture shapes. At the TP3 per-rank shapes the small linears fall well below
it. A shape of 1.3 MB costs 16 us, so a fixed cost of roughly 15 us a launch is visible; that is a reading of the
table, not a measured launch overhead.

## 3. Bound on what this could be worth

Per decode step on rank 0: 64 MLP linears (gate/up and down), 48 delta-net layers (in and out), 16 attention
layers (q/gate, k, v, out); 288 GEMM launches, 112 of them under 16 MB. Summed from the table: **14.3 ms of GEMM
time against a 22.2 ms decode step** (one stream, 3 speculative tokens, the client-side step time of one run; the
layout comment records 22.4 ms), about 65%. The remaining 8 ms is the difference, not a measurement.

If every sub-target row reached a given rate (k and v excluded as launch-floor bound):

| target rate | time saved per step | share of 22.2 ms |
|---|---|---|
| 650 GB/s | 0.88 ms | 4.0% |
| 751 GB/s (the gate/up row's rate) | 1.84 ms | 8.3% |

Per-row at 751 GB/s: MLP down 0.63 ms, delta-net out 0.57 ms, delta-net in 0.36 ms, attention out 0.17 ms,
attention q/gate 0.11 ms. Both bounds are above the 0.5 ms floor set for a patch, so the small-shape decode rows are
worth a measurement plan; they are an upper bound. Caveats: isolated cold single-card rows, not a recorded-pass
replay; the three ranks run concurrently and the slowest sets the step; the R9700's shapes and rows were not
measured here.

Hypothesis, not a finding: the small shapes are launch- or tail-bound (5120 x 1536 is about 8 us of bytes against
21 us measured), so fewer-wave tiles, split-K, or fusing delta-net out and attention out with their neighbours may
beat tuning the same kernel.

## 4. Gated delta net chunk scan against the third-party library

By source and compiled-resource reading only (no timing). libr4d's gfx1201 chunk scan is one kernel that replaces
three FLA stages and keeps the recurrent state in WMMA accumulators across the chunk loop, so the state, W, U and the
new V never reach memory. The third-party library splits the same prefill scan into more, separate kernels. libr4d
documents its register-pressure choices as measured against the alternatives. Conclusion: nothing to port; our
design is the more fused one. The native gfx11 chunk scan remains shelved for accuracy (KLD 5.52 in the lab
history), so on the 7900 XTX it still runs through the compatibility layer.

## 5. Trace modes that failed (so nobody repeats them)

- `rocprofv3 --attach <pid> --kernel-trace` against the engine started by `fn-profile.sh` on `qwen27b-tp3`: the
  attach library returns status 19 and no trace is written. The container does not allow attaching to a running
  process; the 2026-11-23 kernel census recorded the same ptrace refusal.
- Launch-time `rocprofv3` instrumentation: not retried. The census found it ends in device errors during all-reduce
  and router initialisation, and the risk of wedging the cards is not worth a third attempt.
- The serve path itself was clean: the engine started, answered a 2037-token prompt with 64 tokens (one stream,
  about 51 tok/s client-side, step 22.2 ms), and stopped through the owner's stop with the lease released.

## Not established

Which libr3 row serves each op in the running engine (the engine's startup receipt names the attention rows only);
the step share of attention decode and delta-net decode; the R9700's per-shape table; any effect under graph replay.
The per-step phase trace is used in section 6 (added later the same day).

## Not included

No third-party binary, decompiled source, disassembly or symbol list is in this repository. The analysis artifacts
stay outside it.

## 6. Addendum: step-phase trace and the pin route

### Per-op shares from the engine's own trace

`bc-serve serve qwen27b-tp3 --step-phase-trace` (rad0055), one stream, two repetitions of a 512-token prompt with 192
new tokens; 438 rows a rank, every row `valid`. Tracing synchronises at the end of each pass: the traced step was
36.7 ms against about 22 ms untraced, so read shares, not absolute times. `compute_ms` and `collective_ms` overlap
across lanes and are not used here; the table sums the per-op `tape_segments` of the target-verify pass (draft pass 0,
4 tokens, 137 passes), medians.

| op | rank 0 (7900 XTX) | rank 2 (R9700) | launches a pass |
|---|---|---|---|
| quantised GEMM (`gemm_nt_q`, the IQ8R linears) | 51.5% | 37.0% | 288 |
| `prefetch` | 12.9% | 16.8% | 175 |
| `all_reduce` | 11.2% | 29.4% | 129 |
| delta-net decode update | 5.2% | 3.9% | 48 |
| norm + rotation + int8 quantise | 5.1% | 4.5% | |
| paged attention decode | 4.5% | 0.9% | 16 |
| other GEMMs (`gemm_nt`, `logits_gemm`) | 5.1% | 4.4% | |

- Segment sums are 25.5 ms (rank 0) and 25.3 ms (rank 2) against a median traced pass wall of 31.4 ms. About 6 ms of
  the traced wall is outside every segment (gaps and the end-of-pass synchronisation) and is not attributed.
- The GEMM launch count (288) equals the static count in section 3. The three GEMM entries together are about 14.5 ms
  on rank 0, against 14.3 ms summed from the isolated rows in section 2. Measured against the traced segments, the
  GEMM share is confirmed; the absolute prize in section 3 is unchanged.
- The R9700's all-reduce share is mostly waiting for the XTX ranks (VTP36's finding), not link time.
- `prefetch` is 13-17% of the traced pass on both card types. What the op is has not been established.

### The pin sweep did not test the decode row (corrected in section 7)

`gemm_iq8r_nt_m16`, one XTX, cold, three repeats, the six rank-0 shapes of section 2, wave-count (`--pin-mw`) and
token-tile (`--pin-nt`) pins against stock: 0.0-1.8% at M 4 and 16, 0.2-0.7% at M 32, noise level. Reading the source
afterwards showed why: for an int8 weight at M up to 48 (N below 8192 above that) the launcher takes the K-parallel
tile (`gemm_i8_kpar_kernel`), which `--pin-mw` and `--pin-nt` do not reach; they act on the narrow tile only. So the
sweep says nothing about tile geometry on this row. The result stands only as "those two pins are not a lever here".
Side observation: the same row at M 32 costs 2.4-3.0 times its M 4 time (13312 x 5120: 90.8 us at M 4, 217.5 us at
M 32), the eight-stream draft-3 case.

## 7. Prefetch on and off, and the iq8r M 17-32 gap

### Weight prefetch on versus off (qwen27b-tp3, release 2020.8.0)

What it is, from engine source: a read-only device kernel that loads the leading bytes of the next GEMM's weight into
the Infinity Cache beside a decode all-reduce (and the delta-net out-projection beside the recurrent update). Three
sites a layer, 64 + 63 + 48 = 175 a pass, 12 MB at the first two and up to 8 MB at the third, decode steps of up to 32
rows. The sizes were tuned on MXFP4 at TP2 on two R9700s, not on this configuration.

A/B, untraced, ABBA (on, off, off, on), one launch an arm, three seeds and two repetitions each, 512-token prompt,
192 tokens, engine-reported `decode_ms_per_step` (tokens a step matched between arms at one stream):

| streams | on | off | off minus on |
|---|---|---|---|
| 1 | 21.83 +/- 0.05 ms (n 12) | 22.25 +/- 0.02 ms (n 12) | +0.42 ms (+1.9%) |
| 4 | 26.21 +/- 0.30 ms (n 12) | 26.31 +/- 0.27 ms (n 12) | +0.09 ms (+0.4%, inside the spread) |

Per arm at one stream: 21.80 / 22.26 / 22.25 / 21.87. The off arms declared 175 fewer ops. Prefetch stays on; no size
sweep was run (the possible gain is under 2% at one stream).

### iq8r against i8a8 at M 32 (7900 XTX, 13312 x 5120, three repeats, spread under 1%)

| row | warm | cold (own weight copy a launch) |
|---|---|---|
| `gemm_i8a8_nt_m16` | 106.3-106.8 us | 134.0 us |
| `gemm_iq8r_nt_m16` | 169.7-173.4 us | 217.0-217.3 us |
| `gemm_iq8r_tiled` | 272 us | not run |

At M 4 the two rows are equal (26.0 against 26.4 us at 5120 x 2304). At M 32 iq8r is 60-62% slower than i8a8 and does
25 against 41 TFLOP/s; it sits 2.4 times above its memory floor and about 5 times above its compute floor.

### Cause: register spills in the K-parallel tile at NT 2 and above with 256-wide groups

Both rows use the same K-parallel kernel (`gemm_i8_kpar_kernel<KW, SB, NT>`: a block is one 16-row weight tile, KW
waves stride K by scale group, NT token tiles of 16). iq8r runs it at SB 256 (a group is two 128-K units, the inner
unit loop unrolled twice), i8a8 at SB 128. The kernel's resource record in the gfx1100 build, KW 4, unchained:

| NT (tokens) | SB 128 (i8a8): VGPRs, spilled | SB 256 (iq8r): VGPRs, spilled, scratch |
|---|---|---|
| 1 (M 1-16) | 148, 0 | 208, 0, 0 B |
| 2 (M 17-32) | 221, 0 | 240, **127**, 336 B |
| 3 (M 33-48) | 240, 2 | 240, **267**, 816 B |
| 4 (M 49-64, N < 8192) | 240, 35 | 240, **471**, 1584 B |

The pinned occupancy (`R4D_WAVES(6)`) leaves 240 registers a lane; the 256-wide group holds two units' operand
fragments and the next unit's prefetch live at once, which fits at NT 1 and spills from NT 2. That matches the
measurements: equal at M <= 16, +62% at M 32. The gfx1201 build was not inspected.

### Design to close it (not started)

The contract: iq8r outputs stay bit-identical. They can, because a group's int32 sums are exact (|sum| <= 256 x 127 x
127 < 2^24) and the fold order per group and token tile is unchanged by anything below.

1. Cut the live state of the SB 256 body at NT >= 2 without changing arithmetic: do not unroll the unit loop
   (`#pragma unroll 1` on the two-unit loop) and/or prefetch the next unit's weight and token fragments one token tile
   at a time. Expected: the NT 2 form reaches the SB 128 form's resource shape (about 220 VGPRs, no spills), so M 17-32
   runs at i8a8's time (13312 x 5120 cold 217 -> about 134 us at best, -38%). That is a bound, not a promise: the
   per-group fold and the two-unit group still do the same work.
2. If the first form cannot reach zero spills at NT 2: run the NT >= 2 case as NT 1 passes over the same weight tile
   inside one block (the weight fragment re-read from cache), bit-identical by construction, at the cost of extra
   weight-fragment loads.
3. Delivery: a libr3 component rewrite of the shared GEMM source (the existing native override for that file is the
   place), not a vendor edit. Check: selftest bit comparison of the new row against the current one at every M 1-64 on
   the six shapes, then the kernel replay at M 17-32 on the rank-0 shapes with cold rotation, then a decode ms/step
   A/B at 5-8 streams and a KLD run.

Reach: only M 17-48 changes, that is five to eight streams of draft 3 (M = 4 x streams); one and four streams (M 4 and
16) are NT 1 and unchanged. Summed over the rank-0 linears at M 32 (isolated, cold, section 2 shapes) the GEMM time is
about 34.7 ms a step; the i8a8-rate bound of -38% is measured only at 13312 x 5120, so the whole-step bound needs the
other five shapes measured first.
