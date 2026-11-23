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

### Whole-step bound and the gfx1201 record

Selftest rows (`--perfw8 N K M 30 --cold 256`, one XTX, own-weight-copy lines, two repeats), the six rank-0 shapes
summed over a decode step with 64 gate/up, 64 down, 48 delta-net in, 48 delta-net out, 16 attention q/gate and 16
attention out launches:

| M | iq8r (ms a step) | i8a8 (ms a step) | bound if iq8r reached the i8a8 row |
|---|---|---|---|
| 20 | 34.55 | 19.67 | 14.9 ms |
| 24 | 34.60 | 19.72 | 14.9 ms |
| 28 | 34.71 | 19.69 | 15.0 ms |
| 32 | 34.77 | 19.78 | 15.0 ms |

Per shape at M 32 (iq8r against i8a8, us): 13312 x 5120 217.8 / 134.6; 5120 x 6656 148.9 / 79.6; 6144 x 5120 128.3 /
67.4; 5120 x 2304 68.8 / 37.2; 3072 x 5120 66.0 / 37.3; 5120 x 1536 49.0 / 28.1. Every shape is 1.6-1.9 times, so the
gap is not one shape's. The bound is flat across M 20-32 because the cost is the spill, not M. It is an upper bound from
isolated cold rows of one rank; the two rows read slightly different scale bytes; no step time at five to eight streams
was measured here, so it is not stated as a share of the step (the decode A/B will give it).

gfx1201 (the libr4d code object, resource record only): the same kernel has no spills in any four-wave instantiation
(SB 256 at NT 1, 2, 3, 4: 150, 197, 210, 240 VGPRs, 0 spilled). The problem is gfx1100's, from its wider WMMA operands,
so a fix belongs in libr3 only.

### The fix in the engine: 2020.8.1 against 2020.8.0 (decode)

Release 2020.8.1 carries the one-line libr3 change that removes the K-parallel tile's spills at two or more token tiles
with 256-wide groups. Between the two releases the engine binary, libr4d and the config tree are identical; only the
gfx1100 kernel library differs. `qwen27b-tp3` as shipped, untraced, A B B A, one launch an arm, three seeds and two
repetitions an arm (n 12 a release), a 512-token prompt and 192 tokens, engine-reported values, mean and standard
deviation over the 12:

| streams | 2020.8.0 ms/step | 2020.8.1 ms/step | change | engine decode tok/s |
|---|---|---|---|---|
| 6 | 43.79 +/- 2.10 | 36.69 +/- 2.86 | -7.10 ms (-16.2%) | 331 to 418 (+26.0%) |
| 8 | 48.21 +/- 2.76 | 37.67 +/- 0.80 | -10.54 ms (-21.9%) | 383 to 485 (+26.6%) |

Per arm (A1, B1, B2, A2): 6 streams 44.04 / 37.32 / 36.06 / 43.53; 8 streams 48.52 / 37.95 / 37.39 / 47.91. Tokens a
step: 14.5 against 15.3 at six streams (acceptance differs by seed, the same seeds in both arms) and 18.4 against 18.3
at eight, so the gain is the step, not acceptance.

Prefill, separately: median time to first token is level in every arm (six streams about 1030-1075 ms; two early rounds
of each 2020.8.1 arm read 1250-1300 ms; the eight-stream figure is bimodal, 1.2 or 2.2 s, in one 2020.8.0 arm), and the
slowest stream's time to first token is level or lower on 2020.8.1. Read as unchanged, as expected: prefill rows use the
tiled path above 48 rows. No dedicated prefill run.

Against the bound: the isolated bound was about 15 ms a step; the engine took 7.1 ms at six streams and 10.5 at eight
(about half to 70%). Sample: n 12 a release a row, one launch an arm, the same seeds in both releases. Prefill was not
separately tested. Five and seven streams were not run. KLD was not run: the rewrite is byte-identical at every M
1-64, so KLD would be redundant.

## 8. Spill scan, decode attention at one KV head, and the drafter's 4-bit row

Method as above (`r3_selftest`-style replay out of tree, `--cold 256`, median of the launches, each launch timed alone,
one 7900 XTX with the shipped libr3 and the R9700 with libr4d; the replay harness was a scratch copy of the selftest
source with two added perf functions, nothing committed). Releases: 2020.8.1.

### 8.1 Static spill scan

Every AMDGPU code object of the shipped libr3.so (gfx1100) and libr4d.so (gfx1201), resource notes read with
`llvm-readobj --notes` (vgpr_count, vgpr_spill_count, private_segment_fixed_size):

| object | kernels | spilling | max VGPRs spilled |
|---|---|---|---|
| libr3.so, gfx1100 | 7952 | 2009 | 14818 (vocab sort, not a decode kernel) |
| libr4d.so, gfx1201 | 7885 | 789 | 2037 |

Largest families on gfx1100 (instantiations / spilling / max spilled VGPRs / max private bytes), decode-relevant first:

| family | inst | spilling | max spill | max private B |
|---|---|---|---|---|
| iqr (m16, m64, gemv, gated, tiled) | 948 | 546 | 1310 | 3640 |
| r4d_gemm_mxfp4a8_decode_kernel | 280 | 240 | 733 | 2280 |
| r4d_gemm_mxfp4a8_gemv_kernel | 18 | 6 | 3381 | 2076 |
| r4d_gemm_w4a8_nt_m64 (and chained) | 64 | 38 | 1662 | 3176 |
| r4d_gemm_w8a8_tiled / w4a8_tiled / w4a8_prefill | 45 | 45 | 1755 | 5004 |
| r4d_attn_decode_kernel (and chained) | 388 | 258 | 339 | 940 |
| r4d_attn_prefill_kernel | 91 | 77 | 717 | 904 |
| r4d_gdn_chunk_scan_kernel | 8 | 8 | 309 | 920 |

A spill is not a defect by itself; the table is a list of where to look. Reach (does a served row use the instantiation)
is what decides, and the two cases below were checked for it.

### 8.2 Decode attention at the rank-0 shape (one KV head), XTX against R9700

Row `attn_decode_h256_gqa6_fp8kv` (the engine's), q_len 4, one sequence, per call = decode kernel + combine kernel. The
instantiation the engine launches (kernel trace): `r4d_attn_decode_kernel<2,16,256,6,16,0,3429947,0,0u>`, 256 VGPRs,
240 B private scratch; the combine kernel is 24 VGPRs, no scratch. The libr3 native override covers only the paged
prefill kernel; decode is libr4d's source compiled for gfx1100.

us per call, default splits, same KV heads on both cards:

| KV heads | context | XTX | R9700 | ratio |
|---|---|---|---|---|
| 1 | 700 | 87.9 | 17.8 | 4.95x |
| 1 | 8192 | 105.9 | 28.6 | 3.70x |
| 1 | 32768 | 211.8 | 49.0 | 4.32x |
| 2 | 700 | 87.8 | 17.7 | 4.97x |
| 2 | 8192 | 135.7 | 34.7 | 3.91x |
| 2 | 32768 | 362.7 | 74.8 | 4.85x |

At each card's best split count the ratio is 3.85x (700), 4.1x (8k), 4.3x (32k); the bandwidth ratio is 1.5x in the
XTX's favour. Splits pinned (1 KV head, XTX / R9700): ctx 700 default (22 segments) 87.9 / 17.8, 64 or more segments
63.7 / 16.5; at 8k and 32k the default (256) is already the best point on both cards (one segment: 8.2 ms / 0.84 ms at
8k, 32.6 / 3.26 ms at 32k).

Reading:
- The split law's `tiles / 2` cap costs the XTX about 24 us a call at ctx 700 (a pin at 64 or more is 28% faster there,
  7% faster on the R9700): about 0.4 ms a step over 16 layers, nothing at depth. A derivation issue; any fix must derive
  the segment count from the device and geometry, not a new literal.
- The larger part is the kernel's per-workgroup throughput on gfx1100: one segment over 32k tokens takes 10x as long on
  the XTX, and at the best split the XTX moves 16 MiB in 212 us (79 GB/s, 8% of its bandwidth) where the R9700 reaches
  343 GB/s. The bf16-KV row (no fp8 widen) is still 3.1x / 2.0x / 2.5x slower than the R9700 (58.5 / 77.0 / 194 us against
  18.8 / 38.2 / 76.6 at 700 / 8k / 32k), and fp8 costs the XTX a further 1.1-1.5x that the R9700 does not pay. Spills,
  the gfx11 WMMA shim (8-register operands) and the software fp8 widen all contribute; ranking them needs a spill-free
  variant of the decode unit, which was not built.
- Consistency with the engine: 16 layers x 87.9 us = 1.41 ms against 16 x 17.8 = 0.28 ms; the perturbed trace read 1.16
  and 0.22 ms a pass.
- Verify attention does not grow with the row count (1.18 / 1.26 / 1.54 ms a pass at 16 / 24 / 32 rows on rank 0): the gap
  is a constant per pass.
Bound (one stream, one pass a step, if the XTX reached 1.5x the R9700's time): about 1.0 ms a step at ctx 700 and 8k,
about 2.2 ms at 32k; the split-law part alone about 0.4 ms at ctx 700. Not pursued now (a gfx11 attention variant is a
larger project).

### 8.3 The DFlash2 drafter's linears: the 4-bit table row at M 17-32

Form (container descriptor, `rad-info`): `affine codes=u4, scale=bf16[1x128], table=i8{1x16}, fwht128` on every drafter
linear (q 4096x5120, k and v 1024x5120, out 5120x4096, gate_up 34816x5120, down 5120x17408 in 5 layers; fc 5120x25600
once): the IQR `kTable` form, 4 bits, a bf16 scale per (row, 128). Not a `tiles_form`, so `m64_launch` serves M <= 16
with the spill-free 16-token tile and M 17-64 with `iqr_m64_kernel` (nt = ceil(M / 16)). The drafter runs 4 rows a
stream, so five to eight streams are M 20-32.

Resource record, gfx1100, kTable 4 bits, unchained (VGPRs / spilled / private bytes):

| nt | MW 4 | MW 8 |
|---|---|---|
| 1 | 240 / 64 / 164 | 256 / 47 / 96 |
| 2 | 240 / 357 / 1052 | 256 / 298 / 932 |
| 3 | 240 / 750 / 2280 | 256 / 693 / 2180 |
| 4 | 256 / 1020 / 3132 | 256 / 970 / 3140 |

Cold replay of `gemm_iqr_nt_m64` (dtype i8a8h) on the XTX, us, M 16 / 20 / 24 / 28 / 32:

| shape (N x K) | M16 | M20 | M24 | M28 | M32 |
|---|---|---|---|---|---|
| 34816 x 5120 gate_up | 214 | 1709 | 1706 | 1712 | 1687 |
| 5120 x 17408 down | 129 | 971 | 973 | 981 | 974 |
| 4096 x 5120 q | 41 | 287 | 289 | 293 | 297 |
| 5120 x 4096 out | 40 | 245 | 248 | 251 | 253 |
| 1024 x 5120 k, v | 18 | 98 | 98 | 100 | 104 |
| 5120 x 25600 fc | 180 | 1403 | 1405 | 1413 | 1403 |

On the R9700 (libr4d) the same row is flat: gate_up 165 / 189 / 191 us at M 16 / 24 / 32, down 85 / 124 / 124.

Engine reach (step-phase trace, 2020.8.1, six and eight streams, one launch; the draft model pass, compute ms
median / wall ms by rows a pass):

| rows | 4 | 8 | 16 | 24 | 28 | 32 |
|---|---|---|---|---|---|---|
| XTX ranks 0, 1 | 2.86 / 4.6 | 3.18 / 4.9 | 3.28 / 5.2 | 10.2 / 12.2 | 10.4 / 12.3 | 10.5 / 12.5 |
| R9700 rank 2 | 2.33 / 4.6 | 2.46 / 4.9 | 2.34 / 5.2 | 2.36 / 12.2 | 2.43 / 12.2 | 2.42 / 12.5 |

The drafter runs on all three ranks at near-equal time at M <= 16 (replicated, not sharded; the isolated full-shape
M16 total is 2.3 ms). Past 16 rows the XTX ranks' pass grows by about 6.9 ms and the step waits for them. The isolated
figure (17.0 ms against 2.3 ms a pass at full shapes, up to 14.7 ms a step) is an upper bound the engine does not reach
(+6.9 ms); the traced step is 51.4 ms at six streams and 57.0 ms at eight against 36.7 / 37.7 ms untraced (the trace
perturbs), so scaling the excess by 37.7 / 57 gives about 4.6 ms of the untraced step (~12%) as the realistic ceiling for
a fix. Sample: one launch, 49 passes at 24 rows and 47 at 32; the trace perturbs timing, so shares are not additive.

Design of a fix (not done): the same family as the iq8r one, but not a single pragma. In `iqr_m64_body` the unit loop
(`kU = FK / kBlock = 2`) is unrolled and `pf[]`, `pfn[]` (this and the next chunk's weight codes), `bf[]`, `raw[]`, `out[]`
are indexed by the unit or token tile, so `unroll 1` would move them to private memory. Candidate cuts that leave the
arithmetic and fold order untouched, in order: (a) issue the next chunk's fetch after the unit loop so `pf` and `pfn`
are not both live; (b) hold the staged activation rows across fewer rows; (c) run nt >= 2 as nt 1 passes over the same
decoded weight tile inside one block. Contract: byte-identical output at every M 1-64.

### 8.4 What else grows past 16 rows in the target pass

Target verify pass (draft_pass 0), per-op tape segments, medians (23 / 47 / 45 passes at 16 / 24 / 32 rows), ms a pass,
XTX rank 0 | R9700 rank 2:

| op | 16 | 24 | 32 | R9700 16 / 24 / 32 |
|---|---|---|---|---|
| gemm_nt_q (288 launches) | 14.00 | 20.25 | 20.73 | 9.70 / 10.67 / 10.82 |
| gdn_conv_recurrent_update_had | 1.61 | 2.36 | 2.80 | 1.44 / 1.87 / 2.23 |
| attn_paged | 1.18 | 1.26 | 1.54 | 0.31 / 0.33 / 0.37 |
| logits_gemm | 0.63 | 0.92 | 0.97 | 0.72 / 0.73 / 0.75 |
| all_reduce (wait) | 5.63 | 7.21 | 9.32 | 10.47 / 18.01 / 20.66 |

Compute on rank 0 grows 23.95 -> 30.64 ms (+6.7) and on the R9700 18.90 -> 19.46: the int8 GEMMs explain it (+6.25 ms
against +0.97). Per launch on rank 0: gate_up ~90 us at 16 rows, 125-135 us at 24-32; down ~60 -> ~80 us; the R9700's
launches do not move. After the 2020.8.1 fix the iq8r nt 2 row runs at the i8a8 rate (137 us at M 32 for 13312x5120),
so what remains is nt 2 itself against nt 1, in both rows, with no spill (iq8r nt 2 is spill-free, i8a8 SB128 nt 2 is 221
VGPRs and 0 spills). Excess over the R9700's growth: about 5.3 ms of traced compute, about 3.5 ms of the untraced step
at six to eight streams. The cause is not visible in the resource record; a profile (VALU, WMMA, LDS and occupancy
counters) of the six shapes at M 16 against M 17-32 would show it.

### 8.5 The nt 2 cost: counters unreadable, cost model from the sweeps

A hardware-counter profile of the K-parallel tile (iq8r SB 256 and i8a8 SB 128; 13312x5120 and 5120x6656; M 16, 24, 32) was
attempted on the standalone replay with `rocprofv3 --pmc` (three counter sets, then twelve single counters). Only `SQ_WAVES`
(3328 = 832 blocks x 4 waves) and `SQ_BUSY_CYCLES` (about 11.2 M at nt 1, 14.7 M at nt 2) returned values; occupancy, VALU and LDS
busy, bank conflicts, memory-unit busy, L2 hit/miss and the SQ wait counters read 0 on every kernel. No device error. Kernel
trace of the nt 2 launch: 224 VGPRs, no scratch, 8192 B LDS, 128-thread blocks, 832 blocks: 6 waves a SIMD (nt 1: 7 for iq8r at
200-208 VGPRs, 10 for i8a8 at 148).

The post-iq8r-fix cold sweeps give the shape instead (iq8r and i8a8 within 3%, us):

| N x K | M16 (nt 1) | M20-32 (nt 2) | M40-48 (nt 3) |
|---|---|---|---|
| 13312 x 5120 | 93.4 | 137.3 | 191 |
| 5120 x 6656 | 55.5 | 86.0 | 115 |

Time is linear in the token tiles with a constant: t = 49 + 44 x NT us for the first shape (nt 1 93, nt 2 137, nt 3 181
against 191 measured). The weight stream alone is about 71 us at 960 GB/s and nt 1 already runs at 76% of it; one token tile's
math at 90 TOPS would take 24 us. Each extra token tile instead adds 44 us (about 50 TOPS marginal, 32 TOPS overall at M 32), so the
per-tile WMMA and fold work is added to the weight-stream time inside each wave, not hidden under it.

Design (not done; sound on paper, with three points to measure): run the token tiles in parallel waves instead of looping them
inside a wave, a block of KW x NT waves with wave (w, j) taking K stripe w and token tile j. Each wave then carries the nt 1 live
state; waves of a pair read the same weight fragments (L1/L2, no extra DRAM if it holds); the cross-wave reduction and every
per-token-tile fold keep today's order, so output bytes are identical by construction. Estimated nt 2 row 100-115 us on
13312x5120 (between nt 1's 93 and today's 137), about 2.5-2.8 ms of the untraced step. To measure: effective bytes against the
nt 1 row, the LDS reduction at KW x NT waves, occupancy per instantiation.

### 8.6 Wave-per-token-tile trial rejected; two probes of the nt 2 cost

The wave-per-token-tile design of 8.5 was built and measured (archived: PR 1447). Registers and bit identity passed (VGPRs at
the nt 1 level, no spills, 2,304 of 2,304 outputs identical), but the gates failed: cold effective bandwidth at M 32 was 63-70%
of the nt 1 row (gate 80%), 13312 x 5120 nt 2 took 143-144 us against 137 us (+5%, both rows), nt 3 was slower, and the
small shapes were mixed (-14% to +5%). The LDS-reduction gate was not run. The duplicate weight read therefore costs time; the
cause is not attributable without counters. The fit t = 49 + 44 x NT explains the cost; the fix that follows from it does not work as designed.

Two scratch probes (one XTX, cold, 30 launches, timing only, nothing committed) then tested a second reading, that each block's
re-read of the whole activation (grid N/16 blocks, every block reading the [M, K] token fragments) limits the tile:

| probe | 13312 x 5120 M 4 / 16 | M 20 / 32 (nt 2) | M 48 (nt 3) | 5120 x 6656 M 20 / 32 |
|---|---|---|---|---|
| stock K-parallel tile, i8a8 / iq8r (us) | 91.7 / 91.6, 93.9 / 93.8 | 133.8 / 131.4, 134.2 / 133.3 | 187 / 183 | 79.9 / 83.1 |
| activation row pitch 128 B instead of K (rows overlap; wrong numerics), i8a8 | 91.7, 93.8 | 134.1, 135.1 | 185.6 | 80.0 |
| narrow tile (`r4d_gemm_i8a8_pin_split_waves` 1024 target), i8a8 / iq8r | 105.6 / 103.6, 105.4 / 103.7 | 163 / 159, 167 / 164 | 277 / 265 | 93.8 / 90.9 |

- Collapsing the activation footprint per block (about 12 times fewer distinct bytes) changes nothing at any M or shape (within
  0.5 us except 5120 x 1536, 1-2 us): the activation re-read is not the limit, so a block owning several weight row tiles to
  share it would not help.
- The narrow tile, forced by the split-waves pin (targets 256, 512, 1024 tried), is slower everywhere: at M 20-32 on 13312 x 5120
  it is 159-185 us against 131-134 us, at M 4 and 16 it is 12-17 us slower, and at M 48 much slower. The kernel choice in the launcher is right.
- What stays: nt 2 costs about 44 us a token tile on 13312 x 5120 (iq8r and i8a8 alike), it is not weight DRAM, not activation
  fetch and not block geometry (wave-per-tile moved it by only +5%), and the remaining candidates (per-wave WMMA and fold issue, or LDS) cannot be told apart without counters.

### 8.7 Where the nt 2 cost is: the gfx11 int8 WMMA shim, and a bit-identical fix measured in scratch

Scratch variants of `gemm_i8_kpar_kernel` (SB 128, KW 4, built as a single-unit libr3 for gfx1100; stock rebuilt to the shipped
resource record first; one XTX, cold, median of 30 launches, 2 repeats; timing driver adds a fixed launch overhead of about
10 us to every number, so read differences and ratios). 13312 x 5120, M 32 (nt 2), us: stock 153 (M 16: 104).

| variant (tile 2 = the second token tile) | M 32 |
|---|---|
| tile 2's loads kept, its WMMA removed | 107 |
| tile 2's loads and WMMA removed, fold and LDS reduction kept | 107 |
| tile 2's loads removed, its WMMA kept (own accumulator, rotated operands) | 141 |
| tile 2's fragments prefetched one unit ahead (bit-identical, 186 VGPRs) | 147 |
| the two tiles' WMMA chains interleaved (bit-identical, 172 VGPRs) | 147 |

Reading: the loads, the fold and the LDS reduction of the second tile cost nothing measurable; its WMMA costs about 35 of the 47 us.
Prefetch and interleave move it by 1-4%. What a WMMA costs is the shim, not the matrix instruction: libr3's
`r3_wmma_i32_16x16x16_iu8_w32_gfx12` (engines/radiance/components/libr3/r3_compat.h) turns each gfx12 call into a gfx11 one
with `ds_bpermute` exchanges for the A fragment (2), the B fragment (2) and the accumulator (4), a select per element, and 8 adds
of the exchanged product into the caller's accumulator, every k step. The kernel issues 8 such calls per 128-K unit per token tile.

Candidate fix, bit-identical by construction (only data movement changes, and the int32 sums are exact): accumulate the 8 k
steps of a scale group in the gfx11 accumulator layout (`__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32` with the running
accumulator as C) and convert it to the gfx12 layout once per group, before the fold, instead of once per k step. Scratch
result, same method, outputs compared by checksum against stock at every shape and M below (all identical):

| shape | M 4 | M 16 | M 20 | M 32 | M 48 |
|---|---|---|---|---|---|
| 13312 x 5120, stock -> accumulate once (us) | 103 -> 98 | 105 -> 99 | 150 -> 106 | 148 -> 108 | 196 -> 124 |
| 5120 x 6656 | 63 -> 56 | 64 -> 58 | 95 -> 63 | 94 -> 62 | 125 -> 82 |
| 6144 x 5120 | 56 -> 51 | 57 -> 51 | 84 -> 56 | 85 -> 56 | 113 -> 70 |
| 5120 x 1536 | 27 -> 24 | 26 -> 24 | 34 -> 27 | 35 -> 27 | 43 -> 31 |

Resource record (SB 128, unchained, VGPR / spilled): nt 1 148 -> 141 / 0, nt 2 221 -> 175 / 0, nt 3 240 / 2 -> 240 / 1, nt 4 240 / 35 -> 177 / 0.
The i8a8 entry was timed; iq8r (SB 256) runs the same body and its record moves 200 / 178 / 207 / 231 -> 210 / 172 / 212 / 171
(0 spills throughout), not yet timed. Not shipped: a libr3 rewrite of the kernel body needs its own slice and the checks of
section 7 (bit identity at every M 1-64 on the shapes, chained and unchained, KW 2).

### 8.8 Drafter: two M 16 launches against one M 24-32 launch (the 4-bit row)

From the cold replays of section 8.3 (M 16 uses the spill-free 16-token tile; an M 8 launch costs the same as M 16, weight-bound).
Per drafter pass, 5 layers of gate_up, down, q, out and two kv linears, plus the fc projection once:

| | one launch at M 24 / 32 | two M 16 launches | saving |
|---|---|---|---|
| per layer (us) | 3,410 / 3,420 | 920 | 2,490 / 2,500 |
| 5 layers (ms) | 17.1 | 4.6 | 12.5 |
| plus fc (ms) | 18.5 | 5.0 | 13.5 |

Per shape, two M 16 against one M 24: gate_up 428 vs 1,706 us, down 258 vs 973, q 82 vs 289, out 80 vs 248, kv 37 vs 98, fc 360 vs 1,405.
Isolated figures; the engine's trace (6.6 vs 12.2 ms on the XTX ranks) is the scale that matters. The split would be a launch change
in the row (two calls over rows 0-15 and 16-M-1), no new kernel; per-row bit identity against the stock M 17-64 form is not
guaranteed (different accumulation order) and has not been checked. The four cheap register variants of PR 1448 all still spill at nt 2
(stock 357 / 298 spilled at MW 4 / 8; A 381 / 323, B 417 / 357, C1 412 / 339, C2 401 / 338): the drafter's M 17-32 cost stays open, with structural candidates left
(a smaller K chunk per decode, or this two-launch split).
