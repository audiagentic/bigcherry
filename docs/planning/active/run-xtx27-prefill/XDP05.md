---
id: XDP05
order: 4
plan: run-xtx27-prefill
state: active
created-at: '2026-11-09T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# gfx1100 native W4A4 tiled prefill: LDS reuse and actual-shape A/B

## Why and scope

Continue XDP04 **in the existing `exp/xtx27-prefill-opt` worktree**.
The XDP04 one-wave 16×16 IU4×IU4 kernel proved signed-A4,
unsigned-W4, group128 asymmetric zero-point correction and output
transposition; six small GPU geometries produced CPU-exact outputs.
It is not a useful performance comparison: each separate wave reads
the same A and W repeatedly from global memory and does not distribute
them across output tiles. Here we need an M×N tiled kernel, a bounded
test harness and measurements at Qwen3.8-27B dense projections,
without editing `vendor/radiance` or registering an engine kernel.

**Question to answer:** Can packed native gfx11 IU4 multiplication
provide a meaningful prefill compute or bandwidth advantage when the
complete activation and weight staging/epilogue costs are included?
If no, retain the MXFP4 implementation. If yes, gate promotion on
real-model activation quality/KLD before quantizing BF16 to W4.

## Fixed mathematical contract

```
Aq: int4[-8,7], packed low nibble first, scale_a[M,K/128],
    sum_a[M,K/128] = sum_{k in group} Aq
Wq: uint4[0,15], packed low nibble first, scale_w[N,K/128],
    zero_w[N,K/128] ∈ [0,15]
D[m,n] = Σ_g float32(sum_k int32(Aq[m,k])*int32(Wq[n,k])
                     - int32(zero_w[n,g])*sum_a[m,g])
           * scale_a[m,g] * scale_w[n,g]
C: f32[M,N] for isolated numerical proof (no BF16 rounding yet)
```

No FP8/MXFP4 decoder is involved, so time saved by INT4 WMMA must
outweigh **new** A4 quantization, its scales/sums, and any container
or runtime transformations. The XDP03 stored W4 form has a different
nibble/scale layout: this prototype must **not** claim zero-cost
compatibility with `rad-convert` output.

## Architecture and decisions

### Tile geometry, workgroup ownership, instruction layout

- Start with **2×4 wave32** per block (256 threads), each wave owns
  one native 16×16 output matrix; 32 rows of A × 64 rows of W per
  block => logical 32×64 output tile. `wave=threadIdx.x/32`,
  `wm=wave/4`, `wn=wave%4`, `lane=threadIdx.x&31`.
- On gfx1100 a native 16×16×16 `iu4` instruction consumes **2×i32
  (16 nibbles) per lane**, with lanes 0..15 and 16..31 duplicating
  operand rows. W must be first operand (GPU C rows=W output units),
  A second (GPU C columns=activation rows). Register `e` and lane
  contain a GPU-C entry `[2e + lane/16, lane%16]`. Store transpose
  to the host-visible `C[m,n]`; *never* assume gfx12 output layout.
- Keep accumulators in exact int32 over a single 128-wide scale
  group, eight IU4 WMMAs per group; then subtract W zero×A code sum,
  fold into f32 output once per group. Int32 overflow is impossible
  at K=128 with these nibble bounds: max absolute per-group raw sum
  128×8×15=15,360, plus bounded correction.
- Cooperatively stage A(32×64 B) and W(64×64 B), total
  **6,144 B group working set**, in workgroup LDS (plus optional
  padding). 256 threads can copy contiguous aligned bytes. Barriers
  before consumers and before LDS reuse for the next group.
  This gives fourfold A reuse and twofold W reuse across waves,
  reducing direct global row-strided fetches inside each WMMA.
- **Do not store the two halves of a wave32 as two different A/W
  rows.** This breaks IU4 fragment mapping and corrupted the original
  XDP04 off-diagonal outputs. Reuse each row for both halves.
- Initial implementation prioritizes correctness, simple barriers
  and portability to pinned ROCm. No asynchronous overlap, double
  buffering, persistent blocks, graph capture, or dynamic dispatch
  until profiling proves it matters.

### Packed source layout: explicit costs

A and W originally appear in row-major nibble layout, 64 B per
128-element group per row. For reproducible, coalesced staging,
the experimental **tile-major operand** is:

```
A_tile[(m_tile*groups + g)][(step*16+row)*8 + byte]
W_tile[(n_tile*groups + g)][(step*16+row)*8 + byte]
tile=16 rows, step=0..7, byte=0..7
```

That puts 16 neighbouring rows' 8-byte operands together for each
matrix step. A pair of 16-row A tiles and four 16-row W tiles feed
the full 32×64 workgroup. Host prepacking in a standalone benchmark
is an **offline/test aid**, not a free operation in serving.
Production A must be quantized and written into the correct order
by a GPU producer, with its own kernel or fused producer.
Production W conversion should emit or repack into this layout
*once* per immutable model artifact, and its persistent footprint,
alignment and residency must be counted. The benchmark must
report whether packing and activation quantization were excluded.

### Benchmark validity and risk controls

- Keep XDP04 untiled oracle unchanged as independent instruction
  correctness provenance; compare a new tiled kernel with a
  separately implemented CPU exact integer oracle over small and
  sampled real 27B-shaped outputs. Validate tile boundaries,
  multiple G=K/128, nontrivial zero points, mixed signs, edge rows
  and seeds. Do not rely on one diagonal or a single random output.
- For M512/N17408/K5120, **do not compute the entire** 45.6
  billion-product CPU reference. Compare 128+ independently
  selected cells: tile corners, per-wave borders, first/last
  rows and columns, seeded randoms. Always scan the entire output
  for non-finite values; optionally cross-check selected tiles with
  the XDP04 oracle or a small GPU reference. The sample is reported
  explicitly; a sampled check is not a full correctness proof.
- Time with warmup and device events for **kernel alone**, and
  separately measure (or explicitly flag missing) A4 GPU
  quantization, packing and engine overhead. Guard against warm
  L2, clock variation, single-card interference and outliers by
  ≥3 repetitions/rotated order; record min/median, not one minimum.
- Compare the **same gfx1100 GPU** against a pinned
  `r4d_selftest --perfmxd 17408 5120 512 30` MXFP4 benchmark
  where available (with identical M,N,K and GPU lease). These
  execute different encodings and outputs; **not an equal-quality
  end-to-end comparison**. Prior MXFP4 tile result 931.6 µs
  is a historical reference only, not a contemporary matched run.
- Unoptimized tiled kernel's timings cannot establish shipping
  headroom. Record occupancy, VGPR/LDS pressure and ISA
  instruction if it becomes competitive. A bad result should
  lead to a documented **stop/deprioritize** decision, not
  automatic iteration without numerical evidence.

## Interface, artifact and safety decision

Extend existing `tools/lab/radiance/xtx27_w4a4.py` by
`--kernel {oracle,tiled}` (default oracle), preserving all XDP04
commands, defaults and tests. `--phase compile --run` compiles the
selected standalone HIP source to a separate scratch binary and
digest-bound manifest; `--phase gpu --run` uses the **same single**
`agent-lease acquire gpu --ttl 30m -- ...` wrapper and selects
`--suite` small cases or `--shape-27b` for exact
M512,N17408,K5120. Keep all logs/builds under
`${BC_WORK_ROOT}/xtx27-prefill/w4a4`; no new worktree, model
conversion or production modifications.

If an existing GPU lease is held, perform all card-free design,
static and compile work first, and wait for lawful exclusive
measurement access. Do not run GPU tests without the lease, or
interrupt someone else's job.

## Phased acceptance and explicit promotion gates

| Gate | Required outcome | Production implication |
|---|---|---|
| T1 | HIP compile, code/style tests and static packing-layout proof | Research only |
| T2 | Tiled GPU six-case oracle + multi-group and nontrivial zeros; CPU exact | Research only |
| T3 | M512×N17408×K5120 benchmark with 128+ sampled oracle outputs; no NaN | Speed feasibility only |
| T4 | Same-card three-rep W4A4 vs pinned MXFP4 at matched shapes, include conversion/quant cost in estimated effective throughput | Candidate only if competitive |
| Q1 | Real BF16 model layer activations, outlier-aware per-group A4 vs A8 and MXFP4, activation distribution KL/error | Still no serving |
| Q2 | Matched BF16-origin checkpoint and full-model KL/top1/PPL/exact TP2, VRAM/KV | Only then consider registered kernel |
| R1 | Managed `libr3` implementation + fallback, ownership and admission tests, TP2 wire and cold/warm TTFT | Production qualification |

**Quality kill criterion:** XDP04 already measured synthetic
outlier relative-L2 0.374 for W4A4 versus 0.193 for W4A8. No W4A4
candidate should be enabled for the 27B model merely because a
native instruction runs twice as fast. Work should also evaluate
outlier-preserving activation scaling, clipping or mixed precision;
any additional storage/compute is counted.

## Experimental implementation and measured outcomes (2026-11-09)

### Implementation and exactness

Added `tools/lab/radiance/xtx27_w4a4_tiled.hip`, a standalone
HIP program with templated 8-wave workgroups and static, cooperatively
staged packed A4/W4 tiles. No `libr3` kernel has been registered; the
production code remains untouched. The original XDP04 oracle remains
separate and runnable. Extended the **same**
`tools/lab/radiance/xtx27_w4a4.py` CLI with
`--kernel tiled`, `--tile-m {2,4,8}`, and `--shape-27b`.
The default remains XDP04's original oracle and its digest-bound binary.
A tiled compilation generates a separate executable and SHA-protected
manifest, never overwriting the XDP04 control.

The GPU suite covers **nine logical shapes × three tile geometries =
27 correctness cases** under one lease. Every small case was validated
on **every output** against an independent CPU oracle with exact
group-128 INT32 zero correction and f32 scales. This includes
nonmultiples of 16, truncated blocks, multi-group K and M=N=1.
All 27 passed with 0 reported absolute error.

For the 27B-shaped case, the full D tensor has 8,912,896 entries;
the program scans **all entries for nonfinite values**, and compares
**768 independently selected** entries (corners, boundaries, and
seeded random samples) against the host exact integer oracle.
Every sampled entry passed with zero reported absolute error. This is
a **sampled correctness result**, not exhaustive reference comparison
over 45.6 billion integer products. Tile-major input planes are
generated directly on CPU in the experiment: a production GPU
activation quantizer and pack-to-tile layout **do not exist**.

### Shape-matched GPU timings and decision

One RX 7900 XTX `gfx1100`, all arms run under exclusive
`agent-lease`; input M=512, N=17,408, K=5,120 (40 scale groups),
same target GPU, same dimensional work. HIP event times are for the
isolated GEMM kernel with **prepacked device-resident** operands,
no H2D, no A4 quantization, no weight prepacking and no graph/TP
or model-serving overhead.

| Kernel | M×N wave tile | Median kernel time | Numerical gate |
|---|---|---:|---|
| Native IU4×IU4 tiled W4A4 | 2×4 (32×64) | 1,747 µs (median of three independent runs) | 768 sampled / no nonfinite |
| Native IU4×IU4 tiled W4A4 | 4×2 (64×32) | 1,839 µs | 768 sampled / no nonfinite |
| Native IU4×IU4 tiled W4A4 | 8×1 (128×16) | 2,213 µs | 768 sampled / no nonfinite |
| Pinned libr3 MXFP4 tiled control | existing | **923 µs** (median of three independent runs) | its own selftest kernel path |

W4A4 2×4 independent-run medians: **1,743.3 / 1,754.3 /
1,747.0 µs**; MXFP4 control runs: **922.7 / 924.0 / 919.7 µs**.
The closest W4A4 is **1.89× slower** than the MXFP4 control,
**before** costs unique to W4A4. The W4A4 output is f32 and the
pinned MXFP4 benchmark executes a different BF16/e8m0/E2M1
encoding: this is a dimensional, device-matched feasibility
comparison, **not a same-precision model quality or serving A/B**.
It is already unfavorable to W4A4 as a candidate.

Changing the tile to reuse W across 64 or 128 activation rows
**did not improve** latency. 8×1 raised the median by 27% relative
to 2×4; the hoped-for halving of global W reads is not automatically
an end-to-end gain, because 8×1 increases staged A traffic,
per-block staging, LDS costs and register pressure. This is
observed performance, **not** proof of the specific stall source:
there are no per-instruction performance counters in this run.
The runs were executed serially under leases, not simultaneously
on the GPU.

### Compiled ISA and resources (actual gfx1100 code object)

Used `llvm-objdump --offloading` to extract the gfx1100 code object
into scratch and `llvm-objdump --disassemble` to verify the emitted
`v_wmma_i32_16x16x16_iu4` instruction; it is **not** emulated via
IU8 or widened BF16. `llvm-readelf -n` reports:

| Geometry | LDS/group fixed | VGPR | SGPR | Reported spills |
|---|---:|---:|---:|---:|
| 2×4 | 6,144 B | 87 | 45 | 0 |
| 4×2 | 6,144 B | 86 | 48 | 0 |
| 8×1 | 9,216 B | 91 | 53 | 0 |

The compiled code contains native `s_barrier` synchronization
around the cooperative staging and consumption cycle. A scale
group resets int32 accumulators, runs eight IU4 WMMA instructions
and folds eight f32 results per participating lane; there are 40
groups at the 27B shape. Register spills are not the immediate
explanation for the observed regression. Further performance
attribution would need counter-based profiling and code-level
ablations, not a speculative optimization claim.

### Reproducibility

```sh
python3 tools/lab/radiance/xtx27_w4a4.py --phase plan \
  --kernel tiled --suite
python3 tools/lab/radiance/xtx27_w4a4.py --phase compile \
  --kernel tiled --run
python3 tools/lab/radiance/xtx27_w4a4.py --phase gpu \
  --kernel tiled --suite --run --device 0
python3 tools/lab/radiance/xtx27_w4a4.py --phase gpu \
  --kernel tiled --shape-27b --tile-m 2 --run --device 0
python3 tools/lab/radiance/xtx27_w4a4.py --phase gpu \
  --kernel tiled --shape-27b --tile-m 4 --run --device 0
python3 tools/lab/radiance/xtx27_w4a4.py --phase gpu \
  --kernel tiled --shape-27b --tile-m 8 --run --device 0
```

All GPU invocations acquire and release the exclusive lease, fail
closed on an oracle mismatch and record logs in
`${BC_WORK_ROOT}/xtx27-prefill/w4a4`.
The pinned MXFP4 control was `r4d_selftest <libr3.so>
--perfmxd 17408 5120 512 30` from the existing Radiance build,
also under an exclusive GPU lease. Raw logs:
`gpu-tiled-suite.log`, `gpu-tiled-27b-tile2.log`,
`gpu-tiled-27b-tile4.log`, `gpu-tiled-27b-tile8.log`,
`tiled-2x4-27b-repeat2.log`,
`tiled-2x4-27b-repeat3.log`,
`mxfp4-gfx11-27b-repeat{1,2,3}.log`,
and the compiled object `gfx1100-w4a4-tiled.0.hipv4-*`.
No new worktree, model artifact or GPU allocation persists.

### Decision record: NO-GO for serving integration

**Decision XDP05-D1 (measured):** do not integrate this W4A4
kernel into `libr3`, do not convert the BF16 checkpoint to broad
W4A4, and do not use these timings in XDP03's policy selector.
Correct native IU4 arithmetic is not sufficient: the cheapest
tile is still 1.89× slower than the already-supported MXFP4
tiled GEMM *even with prepacked inputs*. The XDP04 synthetic
outlier quantization screen also showed 0.374 relative L2 W4A4
vs 0.193 for W4A8; real-model KL is unmeasured and potentially
worse. The current 4-bit model conversion plans also save only
~70 MiB versus MXFP4.

**Decision XDP05-D2 (architecture):** retain XDP04 and XDP05
as a correctness and layout fixture for future signed/unsigned
int4 work; do not rewrite upstream `vendor/radiance`. Future IU4
proposals must supply a specific *measured* mechanism that cuts
both staging and group-scale overhead enough to beat 923 µs
**including** production A4 quantization and activation quality.
Changing WM/WN alone does not meet that bar.

**Decision XDP05-D3 (priority):** next profiling/engineering
effort returns to XDP01's already-active MXFP4 GEMM path and TP
all-reduce/CPU coordination costs. XDP01 per-op diagnosis found
~2,311 ms `gemm_nt_q` and ~821 ms `all_reduce` across the
profiled requests; those are diagnostic totals, not a serving
speed score. A dedicated correctness and paired TP2 A/B gate is
necessary before any wire/collective optimization is promoted.
Original Qwen3.8-27B context is 262,144 and its future 1M
KV objective remains separate (XDP02).

## Scope clarification after upstream gfx11 research

[XDP06](XDP06.md) compared this proof against published RDNA3 engines. Hipfire already has a native gfx1100 IU4 128x128 output-tile kernel and reports a 1,061-us XTX projection at corresponding dimensions, versus XDP05 1,747 us (cross-project, nonpaired). That reopens **isolated architecture research**, but not production promotion: 1,061 us is still above BigCherry MXFP4 923 us before A4 producer cost, and the quantization layouts/quality differ. XDP05 D1 remains a no-go for *this* tested kernel and broad BF16->W4A4 conversion.

## Progress and decisions

- 2026-11-09: Implemented and verified three tiled HIP variants;
  all 27 suite cases passed; all 768 sampled 27B outputs per variant
  passed. Same-card, actual-shape three-rep native W4A4 median
  1,747 µs vs MXFP4 923 µs: 1.89× slower without A4 quantization.
  Verified genuine gfx11 IU4 instruction and spill-free code object.
  **NO-GO** on conversion or engine integration; focus back on
  active MXFP4 and TP all-reduce (D1-D3 above).
- 2026-11-09: Opened as XDP04 follow-on in the same worktree.
  Selected cooperative 2×4 wave 32×64 output tile, K128 group
  staging into 6KiB LDS, explicit tile-major A/W packing. Chose
  independent CPU oracle and actual-shape sparse verification
  ahead of performance claims, with existing MXFP4 baseline.
