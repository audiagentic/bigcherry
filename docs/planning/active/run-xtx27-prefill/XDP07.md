---
id: XDP07
order: 6
plan: run-xtx27-prefill
state: in_progress
created-at: '2026-11-09T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# gfx1100 INT4 prefill: staged-header root cause and matched GPU experiments

> Archived 2026-11-12 from branch `exp/xtx27-prefill-opt` (git tag `archive/exp-xtx27-prefill-opt`), which never merged. Its lab programs and tests are in that tag.

## Owner objective and constraints

Continue XDP06's empirical comparison with hipfire's native gfx1100
IU4 implementation, while retaining the **existing**
`exp/xtx27-prefill-opt` worktree. Diagnose why BigCherry's
original 32×64 W4A4 prefill tiled kernel was 1,747 µs
at M512/N17408/K5120 versus hipfire's reported ~1,061 µs.
Keep XDP04 and XDP05 reference kernels immutable; source changes
here are experimental only, under
`tools/lab/radiance/xtx27_w4a4_wide16.hip`,
`xtx27_w4a4_wide8.hip` and controlled wrappers.
No `vendor/radiance` edits, model conversion, production wiring,
new worktree, GPU use without a lease, or push/PR.

### Source-informed hypothesis

Read hipfire's MIT-licensed
[direct gfx11 IU4 kernel](https://github.com/warpfront/hipfire/blob/master/kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip)
and its
[gfx1100 negative experiments](https://github.com/warpfront/hipfire/blob/master/docs/plans/2026-09-15-lloyd-gfx11-iu4-split.md).
Hipfire's 128×128 native IU4 path co-stages weight nibble fragments,
half2 weight scale/zero headers and 128-element A4 scale + exact
int32-code-sum sidecars in LDS. Its 16-wave variant uses
32 persistent f32 sums/lane, one live int32 WMMA accumulator chain,
and a compiler scheduling fence between subtiles. It reports
~98 VGPR in its low-live-state experiment and a 1,061-µs
gfx1100 kernel benchmark at the same logical projection shape.
These **are independent hipfire observations**, not BigCherry
measurements or an equivalent original BF16 model-quality A/B.

Our initial 16-wave 128×128 XDP06 kernel had exactly the same
arithmetic formula but read `sa[m,g]`, `sw[n,g]`,
`asum[m,g]` and `wz[n,g]` in every output fold
over 40 K128 groups. Despite caching, this increases address
calculation and VGPR lifetimes. Its direct/coalesced 147/149
VGPR allocations were an independent clue.

## Invariants and implementation

The exact research mathematical contract remains:

```
C[m,n] = Σ_{g=0}^{K/128-1}
         (Σ_{k in g} signed_A4[m,k] * unsigned_W4[n,k]
           − zero_W[n,g] * Σ_{k in g} signed_A4[m,k])
         * scale_A[m,g] * scale_W[n,g]
```

Packed signed/unsigned 4-bit operands, original synthetic RNG seed,
group128 integer correction, f32 `fmaf` order, row-major C and
independent CPU oracle are unchanged.

- **Base 16-wave / 128×128**: two output-store modes, direct scatter
  and final LDS-skewed transpose. Both compiled and passed all
  24 small GPU oracle cases (12 shapes × 2 store modes) and
  768 sampled full-shape outputs/mode. Coalesced stores alone
  improved median **2,310.7 → 1,951.9 µs** (15.5%).
- **`wide16sched`**: separate translation-unit wrapper selects
  `__builtin_amdgcn_sched_barrier(0)` after each output-subtile
  fold. Compiler resources **unchanged** (147/149 VGPR, no spills).
  Negative compiler-only result, deprioritized for hardware.
- **`wide16meta`**: four 128-entry shared arrays hold A f32
  scales, A exact int32 sums, W f32 scales and W int32
  zeros. Four disjoint 128-thread cohorts load one scalar
  each per group, before the same workgroup barrier used
  for packed operands. All later output folds read those
  headers from LDS, with identical formulas and output mapping.
  Native gfx1100 code objects now use **104 VGPR direct,
  110 VGPR coalesced**, ~18 KiB LDS, **zero spills**.
- **`wide8`**: 256-thread version of the same 128×128 output
  tile uses 64 persistent sums/lane and compiled to **165
  VGPR**, no spills. The reduced workgroup size does not
  imply better residency or workload speed.
- **96-VGPR cap negative preflight**: forcing
  `__launch_bounds__(512,2)` and `amdgpu_num_vgpr(96)`
  on `wide16meta` produced **8 VGPR spills and 36 B
  private thread storage** (direct), or **12 VGPR spills
  and 52 B private storage** (coalesced). The scratch-only
  binary is not a selectable GPU experiment.
- **`wide16metatile`**: separate wrapper selecting
  `XDP06_STAGE_META` and `XDP06_META_TILE_MAJOR`.
  The Host constructor and CPU oracle remain **row-major**.
  Only the Device copy of the four metadata planes
  performs an offline host reordering to
  `((row//128)*G+g)*128+(row%128)`.
  The GPU staged header loads become contiguous for
  neighboring rows; invalid tail rows are zero-padded.
  Repacking is excluded from kernel timings, so this
  version carries **extra unpriced producer cost**.
  The code object keeps **104/110 VGPR**, zero spills,
  LDS 18,432/18,560 B, with 4 additional SGPRs.
  The IU4 arithmetic itself is identical.

The single opt-in `xtx27_w4a4.py` runner has distinct
`--kernel` choices for `oracle`, `tiled`,
`wide`, `wide8`, `wide16sched`,
`wide16meta`, and `wide16metatile`.
Each has its own scratch executable, manifest and logs.
Included-source wrappers hash **both the wrapper and
the included base source**, so source edits invalidate
stale binaries. `--phase plan` runs without a
subprocess; `--phase compile --run` is GPU-free;
`--phase gpu --run` acquires the exclusive
`agent-lease`. A mismatch fails closed.

## Measured results: single gfx1100 RX 7900 XTX

All successful runs leased the GPU exclusively, exited
and released the lease. M=512, N=17,408, K=5,120,
40 scale groups, synthetic prepacked weights/activations.
Each process reports median of three HIP-event rounds
(15 iterations/round). Every full-size experimental
mode scans **all 8,912,896 C values** for nonfinite
values and compares 768 stratified/seeded entries
against the *independent* exact-integer CPU oracle;
all passed, with zero reported absolute error.
Every 12-shape × 2-store **24-case** GPU suite
checked all outputs, including tails, and passed.

| Experimental kernel | Full-shape time | Evidence and status |
|---|---:|---|
| XDP05 32×64 W4A4, **fresh control** | 1,762.5 µs | sampled CPU PASS |
| XDP06 16-wave 128×128 direct, before metadata staging | 2,310.7 µs | full-shape PASS |
| XDP06 16-wave 128×128 coalesced, before metadata staging | 1,951.9 µs | full-shape PASS |
| **XDP06 16-wave, metadata staged, coalesced** | **1,056.9 µs** | **median of three independent runs**, full-shape PASS |
| XDP06 16-wave, metadata staged, direct | 1,123.8 µs | median of three runs, PASS |
| XDP06 8-wave 128×128, coalesced | 2,732.5 µs | 24/24 suite + full-shape PASS |
| XDP06 8-wave 128×128, direct | 2,762.1 µs | full-shape PASS |
| XDP06 group-major packed headers, coalesced | 1,083.8 µs | median of two runs, PASS |
| XDP06 group-major packed headers, direct | 1,142.1 µs | median of two runs, PASS |
| Pinned libr3 **MXFP4 tiled, fresh same-XTX control** | **927.9 µs** | own selftest, *different format* |

The staged-metadata coalesced independent medians were
**1,064.0, 1,033.3 and 1,056.9 µs**, with three timed
HIP-event rounds each. Relative to the **same-card**
fresh XDP05 control, this is ~**1.67× faster**.
Relative to the previous 128×128 coalesced 1,952-µs
prototype, it is ~**1.85× faster**. Relative to
the pinned MXFP4 kernel-only control it is still
~**13.9% slower**, *before* A4 activation
quantization. Historical hipfire's 1,061-µs
native IU4 number is unpaired and cannot prove
complete-engine performance parity.

The two group-major packed-header coalesced
medians were **1,087.1 and 1,080.4 µs**,
a two-run median of **1,083.8 µs**. That is
~2.5% worse than the row-major staged-header
three-run median. Even if clock variance explains
some of this difference, there is **no observed
benefit** to justify its extra metadata repacking.
The 8-wave topology lost badly, at ~2.6× the
16-wave staged-header latency.

The three new important observations are:
(1) native IU4 was never the fundamental issue;
(2) header-load placement and related register
live ranges were a major kernel bottleneck;
(3) keeping data contiguous is not enough to
justify offline repacking when already staged
headers are small and cache-resident.

## Decisions, residual risks and next gates

**XDP07-D1 (retain):** keep `wide16meta` with
16 waves, per-group LDS headers and coalesced
output stores as the leading native IU4 research
kernel. It passes the oracle, matches hipfire's
reported kernel-only neighborhood, and has
no reported spills. This is **not** a production
promotion.

**XDP07-D2 (reject):** discard 96-VGPR forced
cap (spills), 8-wave tile (2,732.5 µs),
schedule-fence-alone (no resource benefit),
and group-major header repack (no measured
gain plus unpriced conversion) as near-term
serving choices. Retain research fixtures
for reproducible negative controls.

**XDP07-D3 (quality and cost gate):** the
standalone test uses *prequantized synthetic*
A4/W4. The real Qwen3.8-27B BF16-origin
A4 quantizer (clipping/MSE candidates and
outlier treatment), full-model KL/PPL and
TP2 exact-wire output checks are missing.
Hipfire's reported A4 quantizer overhead
belongs to its engine and cannot be assumed
to match ours. Even an unrealistically zero-cost
A4 producer leaves 1,057 µs above 928 µs
on this matched projection.

**XDP07-D4 (next measured engineering priority):**
do not modify Radiance serving or quantize
the BF16 checkpoint on current evidence.
Either (a) prove an **end-to-end A4 producer +
IU4** candidate better than MXFP4 with
real model quality, or (b) prioritize
existing MXFP4 GEMM, cross-XTX TP
all-reduce and CPU orchestration, which
carry validated serving-performance risk.
A new native IU4 kernel optimization needs
a concrete profiled stall source and
repeatable same-GPU improvement, not another
arbitrary tile change.

### Reproduce the bounded probes

```sh
# Existing XDP worktree; no separate permission per variant.
python3 tools/lab/radiance/xtx27_w4a4.py --phase plan --kernel wide16meta --suite
python3 tools/lab/radiance/xtx27_w4a4.py --phase compile --kernel wide16meta --run
python3 tools/lab/radiance/xtx27_w4a4.py --phase gpu --kernel wide16meta --suite --run
python3 tools/lab/radiance/xtx27_w4a4.py --phase gpu --kernel wide16meta --shape-27b --wide-store paired --run
python3 tools/lab/radiance/xtx27_w4a4.py --phase gpu --kernel wide16metatile --suite --run
python3 tools/lab/radiance/xtx27_w4a4.py --phase gpu --kernel wide16metatile --shape-27b --wide-store paired --run
```

Logs and standalone GPU binaries are under
`${BC_WORK_ROOT}/xtx27-prefill/w4a4`,
not the model store or Git. Evidence includes
`gpu-wide16meta-27b-paired-repeat{1,2}.log`,
`gpu-wide16meta-27b-paired.log`,
`gpu-wide16metatile-27b-paired-repeat1.log`,
`gpu-wide16metatile-27b-paired.log`,
`gpu-wide8-27b-paired.log`,
`gpu-wide16meta-suite.log`,
`gpu-wide16metatile-suite.log`,
`mxfp4-gfx11-xdp06-control1.log` and
`gpu-tiled-27b-tile2.log`.
No full model conversion or production
runtime benchmark was performed in XDP07.
