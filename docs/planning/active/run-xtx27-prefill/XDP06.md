---
id: XDP06
order: 5
plan: run-xtx27-prefill
state: active
created-at: '2026-11-09T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# RDNA3 engine survey and native W4A4 prefill redesign decision

## Owner concern and measured starting point

The owner observed that the XDP05 W4A4 native IU4 GEMM result was
unusually poor despite RDNA3's ~2x native IU4 vs IU8 instruction
roof. We should establish **what prevents the measured kernel from
using this compute** before permanently rejecting INT4 as an
architecture. Keep the existing worktree, preserve XDP04/XDP05
control binaries and source, and do not change `vendor/radiance`,
quantize full BF16 weights or register a production kernel.

At M512 N17408 K5120 on one RX 7900 XTX (gfx1100):
- XDP05 W4A4 tiled 2x4, **1,743 / 1,754 / 1,747 µs** across
  three independent runs, ~52.2 integer TOPS effective;
- XDP05 4x2 ~1,839 µs, 8x1 ~2,213 µs; all pass sampled
  host-int32 oracle, 768 samples, zero reported absolute error;
- pinned `libr3` MXFP4 tiled **922.7 / 924.0 / 919.7 µs**
  across three runs (median **922.7 µs**), not comparable
  model quality or formatting. MXFP4 runs execute different
  matrices/encodings, although dimensions and GPU match;
- source and extracted actual gfx1100 ISA show native
  `v_wmma_i32_16x16x16_iu4`, 6,144 B LDS at 2x4,
  87 VGPR / 45 SGPR, no reported spills.

Arithmetic count = 2*M*N*K = 91,268,055,040 scalar integer
ops. 1,747 µs => ~52.3 TOPS, well below the register-only
~285.8 TOPS measured roof. That gap is **not** itself
evidence of a faulty driver or an automatically faster
INT4 algorithm.

## Source-level suspected bottlenecks, not yet established

1. **Frequent global-workgroup barriers:** for each
   K128 group the XDP05 kernel copies 6 KiB into LDS,
   hits `__syncthreads()`, computes eight WMMA steps
   and applies zero-point and scale, then hits a second
   barrier. There are 40 groups at K5120: 80 barriers,
   40 group setup/epilogues. This can stall all eight
   waves on the slowest one.
2. **Serial scaling and correction:** eight C registers
   per lane per group each perform zero-point subtraction,
   A sum and W zero fetch, A and W scale fetch, floating
   multiplication and FMA. The group128 mathematical
   scale is a hard constraint; combining raw integer
   accumulators across groups without accounting for
   different scales **changes the answer**.
3. **Operand staging pattern:** each group loads 6 KiB
   of tile-major packed W/A, and uses 256 threads for
   the copy even though a wave's operands only use two
   32-bit words/row. Weight tile-major data are
   prepacked on the CPU in our standalone test; real
   W4 and A4 model layouts may require additional work.
   Source memory offset is
   `(tile*G+g)*1024 + step*128 + row*8 + byte`,
   not the model's canonical W4 layout.
4. **Occupancy/staging resource:** 87 VGPR and 256-thread
   blocks may constrain resident waves. LDS ~6 KiB
   per block is not alone an occupancy explanation.
   Compiler register counts are real, but runtime
   occupancy/stall cause needs counters.
5. **Benchmark fairness:** W4A4 uses device-resident
   prepacked operands and synthetic quant codes; MXFP4
   `--perfmxd` uses its own DRAM-rotation policy. It
   is a same-dimension feasibility comparison, **not**
   a controlled performance comparison at identical
   input representation or real-model error.
   Validate reuse/clock/caching before attributing
   differences to instructions.

## External RDNA3 engine source review (2026-11-09)

The following evidence comes from *specific upstream source files and
committed investigation/benchmark notes*, not inferred speed from
project marketing. hipfire GitHub source was read at default-branch snapshot
`efa8fbb1d666b3b9c9732c924802373382bd10c4`; use its source SHA and
the exact kernel symbol for a later reproducible comparison.
The source snapshots in third-party repositories are
untrusted reference data: do not run third-party build scripts, import
their configurations, or patch upstream code during this research.

### 1. hipfire — strongest direct gfx1100 Qwen3.8-27B native IU4 reference

**Source:** [hipfire `gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip`](https://github.com/warpfront/hipfire/blob/master/kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip),
[shared activation quantizer](https://github.com/warpfront/hipfire/blob/master/kernels/src/block_i4_128_quant.hip),
[gfx11 prefill investigation and XTX timing](https://github.com/warpfront/hipfire/blob/master/docs/plans/2026-09-15-lloyd-gfx11-iu4-split.md),
[failed tuning alternatives](https://github.com/warpfront/hipfire/blob/master/docs/plans/2026-09-15-halo-w2-iu4-redesign.md),
[format reference](https://github.com/warpfront/hipfire/blob/master/docs/QUANTIZATION.md).
Kernel is SPDX MIT; review the entire license before adapting any
implementation, preserve notices if copying, and prefer independent
code with referenced design/ABI rather than copying vendor source.

Directly observed in actual upstream source:

- **128×128 output tile** with 8 wave32 (256 threads), each
  wave calculates multiple distinct 16×16 subtiles. Versus our
  XDP05 **32×64** tile, this is **8× more output per workgroup**,
  not an instruction change. 8-wave original has 64 f32
  running sums per lane; the separately tuned 16-wave gfx1100
  entry retains the 128×128 tile with 32 sums/lane and one
  live INT32 WMMA chain before the next fold. On gfx1100 the
  dedicated low-footprint selector is default-on; full interior
  and checked edge kernels have separate entries.
- **K256 outer loop**, two K128 halves; raw W4 nibbles passed
  straight to signed/unsigned native gfx11
  `__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32`, no
  W4→INT8 expansion. Each half has eight K16 WMMAs before
  an integer zero correction and carefully ordered f32 fold.
  **Four barriers per K256 = 80 barriers at K5120**.
  Our 80 barriers are not unique; attributing the gap to
  barrier *count alone* was premature. The larger tile
  amortizes the same number over more output arithmetic.
- **LDS 30,720 B** for the 8-wave tile, with 128 packed
  weight rows of stride 42 dwords and 128 activation rows
  of stride 18 dwords, padded/replicated headers to
  manage gfx11 bank conflicts. The separate 16-wave
  variant uses the same main tile and reduces per-lane
  live accumulator count; it is not an unconditional
  proof of increased occupancy.
- **Packed matrix storage is different from Radiance.**
  hipfire MQ4G256V2 stores 136 bytes per K256
  weight-row group, dual f16 scale/zero headers
  for each K128 half, with optional offline FWHT/AWQ
  transforms. Its prequantized A4 sidecar stores
  72 bytes per 128 values: f32 scale + exact i32
  nibble-code sum + 64 packed bytes. No claim that
  Radiance's `rad-convert` W4 weights can be read
  unchanged by this kernel. Write a separate,
  provenance-bound converter or use an isolated
  in-test packing stage; avoid double weight residency
  when maximizing KV headroom.
- **A4 quality is actively tuned.** The quantizer
  supports per-group clipped candidate MSE searches
  (2-candidate fused producer policy on gfx11 where
  enabled), rather than our simple RTN with max/7.
  Producer fusion carries A4 codes, scales and sums
  into the consumer without a redundant separate
  quantizer; captured/recorded graph paths can use a
  different unfused fallback. This is a new execution
  contract and must be quality-checked against the
  same original BF16 27B weights.
- **Symmetric weight option.** When a verified
  symmetric MQ4 artifact permits it, hipfire rebiases
  the packed weight nibble once, uses signed×signed
  IU4, and reduces the per-output asymmetric zero
  work. Never apply this optimization to asymmetric
  tensors without an exact correction proof.
- **Grid specialization and output stores.**
  Separate guarded tail kernels avoid bounds/address
  work on large interior tiles. The current hipfire
  full-tile store uses buffer-resource addressing to
  avoid constructing many 64-bit per-value pointers.
  Our XDP05 stores 8 f32 elements/lane per group
  through generic addresses; investigate store
  coalescing and generated address ISA instead of
  assuming only WMMA/loads matter.

**Useful traffic calculation:** XDP05 2×4 stages
6,144 B per K128 for 2,048 outputs, or
**6.0 staged input B/output per K256**. hipfire
128×128 staging reads one K256 group of
(128×136 + 2×128×72)=**35,840 B** for 16,384
outputs, or **2.1875 B/output per K256**.
These are *logical source-input staging requests*;
L2 reuse, same-WG broadcasting, output stores and
quantizer traffic are omitted, so do not equate
the ~2.74× ratio with predicted performance.

**Direct XTX measurement in hipfire's recorded failed
Lloyd-split investigation:** same M512 (their N512
batch), weight-row N17408 (their M17408), K5120
`full_set_occ3` uniform native IU4 measured
**1,061 µs** on gfx1100, vs our **1,747 µs** best
W4A4. This is a ~1.65× **cross-project
kernel-only reference**, not a paired control
or a guaranteed port. Their separate profile
lists `quantize_int4_mmq_ds128` at ~75 µs/call
for its own execution regime. Our
pinned MXFP4 target was ~923 µs (different
precision/layout and DRAM rotation). Their
failed split-Lloyd twin was **2,813 µs**,
despite 0 reported spills after iteration,
because doubling the native IU4 WMMAs incurred
an actual ~2.7× cost. Do **not** infer all
additional independent chains help.

Their gfx1151 tuning notes reject larger loader/
prefetch configurations that spilled or reduced
resident WGs; read those as mechanisms, never
as gfx1100 performance proof. Some independent
hipfire gfx1100 benchmark and KLD records exist,
but not the same BF16-origin candidate or our
exact-wire TP2 setup; they do not clear the
BigCherry quality gate.

### 2. Strata-rdna3 — cooperative prefetch, but no gfx1100 W4A4 peer

**Local read-only source:**
`src/prefill/wmma_gemm.cu`, `src/prefill/gemm.cu`,
`src/prefill/moe_mmq.cu`, `src/prefill/moe_f16_common.cuh`
in [Strata-rdna3](https://github.com/nicejunjie/Strata-rdna3).
Its RDNA3 dense kernel uses FP16/BF16 WMMA,
not native IU4 W4A4. It has:

- one-wave 16×16 WMMA with **no LDS** for small M;
- 4-wave 64×64 tiled FP16 with double-buffered
  2-KiB-per-buffer W LDS, 16B coalesced global
  loads, overlapping next-tile prefetch;
- optional 8-wave 128×256 FP16 prefill with
  BK32 and paired A/W LDS, register prefetch
  and grouped tile order to reuse weights.
  The author records gfx1151 tuning numbers,
  not same-shape gfx1100 W4A4 comparisons;
- K-quant MMQ paths from llama.cpp, and FFN
  expert F16 WMMA paths. The file
  `mmq_nvfp4_w4a4.cu` targets NVIDIA
  Blackwell `sm_120a` FP4, **not** RDNA3.

**Transfer decision:** borrow shape-based
dispatch, measured double buffering and
coalesced staged loads; not a claimed RDNA3
native-IU4 implementation or Strata's
Flash-Next ~6K prefill result for a
dense 27B kernel.

### 3. hipEngine, llama.cpp/Lucebox, Quark→SGLang

- [hipEngine RDNA3 tuning guide](https://github.com/shisa-ai/hipEngine/blob/main/docs/RDNA3-TUNING-GUIDE.md):
  measured best practice emphasizes byte
  traffic and coalescing, tile-specific
  launch selection, spill-free ISA, and
  LDS only when it provides reusable data.
  Its RDNA3 tuning guide is a method, not
  a matched native-IU4 speed certificate;
  it is published under a more restrictive
  licensing regime. Use ideas with proper
  attribution and a clean independent
  implementation, **not copied code**.
- [Lucebox](https://github.com/Luce-Org/lucebox)
  / llama.cpp HIP quantized MMQ:
  a documented 48x64/4-warp MMQ tile
  instead of stock 128x128/8 improved
  Qwen3.6-27B DFlash verify throughput
  56.78 to 60.18 tok/s (+6.0%) on gfx1100;
  see https://github.com/Luce-Org/lucebox/blob/main/server/docs/HIP_PERF_PLAN.md .
  These small verifier-row tiles differ from
  M512 dense prefill;
  improving M2–M48 decode by using a
  smaller tile would not justify using
  the same tile for large prefill.
  Obtain full exact kernel source/perf
  before claiming this as a competing W4A4
  implementation.
- [RDNA3 Quark W4A16→SGLang](https://github.com/lawsirlawsir-png/rdna3-quark-w4a16):
  direct gfx1100 Qwen3.8-27B W4A16
  engineering (not native A4), including
  lossless reorder Quark↔GPTQ. Source
  documents Quark packed nibble order
  [0,2,4,6,1,3,5,7] and a different
  physical K/N pack. Source patch 0005
  moves activation sums/weight zero
  corrections to once/group, gaining
  ~9–14% at skinny M in their own setup.
  Retain as a precision-preserving
  **fallback research avenue**, not
  evidence of faster W4A4 prefill.

### Comparison and decision implications

| Candidate | Native gfx11 IU4×IU4 | Relevant M512/K5120 | Useful design feature | BigCherry gate |
|---|---|---|---|---|
| XDP05 baseline | Yes, verified ISA | 1,747 µs | exact integer + f32 oracle | control |
| hipfire | Yes, source+own gfx1100 record | 1,061 µs (unpaired) | 128×128 reuse, 16-wave low-footprint, A4 producer | independent parity/quality and same-card A/B |
| Strata | Not established for gfx11 W4A4 | unqualified | large FP16 WMMA, LDS double-buffer | reproduce in native IU4 |
| hipEngine | No matched native IU4 measurement reviewed | unqualified | format/shape occupancy method | source+benchmark validation |
| Quark/SGLang W4A16 | No; W4A16 is not A4 | different | exact nibble transform, A16 quality | independent format/converter route |

**Decision XDP06-D1:** XDP05's 1.89× MXFP4
regression rejects the *particular 32×64 tiled
prototype*, **not the possibility of useful
native gfx11 IU4**. Hipfire's independent
XTX 1,061-µs record is the first credible
reason to reopen a bounded architecture
experiment. However, even 1,061 µs plus
quantization does **not** yet beat 923-µs
MXFP4 GEMM; a speed-up relative to our
prototype is not a production win.

**Decision XDP06-D2:** prioritize a 128×128
IU4 output tile / 16-wave low-live-acc
research kernel and compare against the
same-data 32×64 XDP05 control. Before
allocating full model memory, examine
workgroup count, operand and output
transaction addresses, barriers, allocated
VGPRs, L2/VMEM and LDS waits. A
2×4→4×2 wave reshuffle at fixed 2,048
outputs has already failed; increasing the
tile by 8× is a new and falsifiable lever.

**Decision XDP06-D3:** keep activation
quantization/quality a *separate and
necessary* gate. A clipped two-candidate
A4 quantizer may reduce error, but it is a
different recipe and must pass real
Qwen3.8 layer activation, BF16-origin KL
and TP2 quality comparison. Do not
silently promote W4A4 based on synthetic
kernel-only speed. Keep XDP03 quant
selection gated on actual data.

## Ablation design and decisions

### A. 128×128 output reuse (new primary test; staged)

Build a **separate** standalone 128×128 native IU4
kernel with the same logical C formula and the same
independent oracle as XDP05. Begin with 16-wave
(512-thread) low-live-accumulator 128×128,
32 output values/lane; next 8-wave 128×128
(64 values/lane) only if the compiler keeps
reasonable VGPR allocation and occupancy.
Keep G128 integer accumulators and folds;
don't duplicate hipfire's external file or
misinterpret its FWHT/MQ4 group layout
as the existing Radiance W4 representation.

The 128×128 variant has **544 logical workgroups**
for M512×N17408, vs XDP05's **4,352**.
Measure the launch-geometry change with
clocks/occupancy and all output validity
invariant. The larger 30KiB LDS budget
and 512-thread launch may reduce workgroup
residency: document actual allocated
VGPR/SGPR, private scratch and 64KiB
LDS constraint. If 16-wave fails the
small full-output oracle, do not benchmark.

Source-level proof before hardware: enumerate
output ownership, two's-complement nibble
mapping, K128 scale/correction order, LDS
publishing/consumption and safe partial
M/N tiles. Check W4 producer layout vs
our own scratch format, quantify extra
offline and runtime packing bytes. Run 27
small full-oracle cases, 768+ sampled full
projection outputs, matched cold/warm
three-rep kernel and same-device MXFP4.
A gain versus XDP05 is a research win;
a production win still demands faster
than 923 µs including A4 quantization
plus model-level KLD.

### B. Direct packed fragment loads, without workgroup LDS/barriers

Build a **separate**, standalone
`xtx27_w4a4_ablate.hip` kernel so XDP05's passing
source, binary and its manifest remain a known-good
control. The direct variant still uses native signed-A4
and unsigned-W4 IU4 WMMA with identical group math,
tile-major host input and 2x4 wave tile.

Each wave directly reads its two packed `uint32`
operands per row and K16 step from device memory,
which may reuse L0/L1/L2 within the workgroup but
duplicates some row loads across the 8 waves.
This **removes both group barriers and LDS staging**,
at the possible cost of increasing aggregate load
transactions/cache pressure. Controlled CPU oracle
equality tells whether reordering is correct; matched
shape GPU timing tells whether bypassing barriers
helps. If faster, optimize global coalescing and
cache/LDS handoff; if slower, retain LDS.

Do not change scale, zeros, seed, output C dtype or
number of WMMA steps. The same `--suite` must
exercise M1, M17/N65, K128/K1024, and N17408/K5120
with CPU verification, always under one exclusive
`agent-lease`. Compile without GPU access while
another agent holds the lease.

### C. Controlled epilogue instrumentation (follow-on)

If larger output tile and direct versus LDS are inconclusive, prepare an
identical-data diagnostic variant that records GPU
performance counters/instruction mix and separates
(i) copy/barrier, (ii) IU4 WMMA and (iii) scale /
zero correction by *timed ablations that preserve
correctness for main benchmark*. Diagnostics that
skip arithmetic must have a different output marker
and cannot be promoted as a real GEMM.

Optional longer-term candidates:
- symmetric signed-W4×signed-A4 to remove
  asymmetric zero point, but recheck weight error,
  activation error, KLD and converter storage;
- fuse GPU A4 quantizer with the producer and pack
  directly to tile-major layout, recording its launch
  and VRAM footprint;
- assign several consecutive K128 groups to a
  software pipeline/double-buffer to reduce stalls
  without combining dissimilar scales;
- tune M/N wave geometry only after the dominant
  stage is identified; XDP05 already found 4x2
  and 8x1 slower.

### D. Acceptance and stop criteria

- At least one independent oracle-backed *new-mode*
  GPU suite, covering tails and multiple groups, must
  pass before any timing is accepted.
- Full 27B shape: all outputs scanned for nonfinite,
  768+ representative outputs compared to independent
  host integer+f32 oracle; measure 3 timed rounds
  per run and ≥3 independent runs for any speedup
  claim. Fresh same-card MXFP4 baseline/temperature
  interleaving if practical. Logs in
  `${BC_WORK_ROOT}/xtx27-prefill/w4a4`.
- **Near-term feasibility:** a new variant needs to
  show measurable gain over XDP05's 1,747 µs.
  **Production bar** is stronger: beat ~923 µs
  *including* activation quantization and packing,
  plus model KLD comparable to MXFP4.
- Never claim same-model quality or convert the
  checkpoint based only on synthetic, prequantized
  4-bit inputs. Preserve XDP05 NO-GO until both
  performance and quality gates pass.
- Every GPU run must acquire an exclusive lease; if a
  peer holds it, perform only card-free design/build
  work. Do not interfere or force another holder's release.

## Progress / decisions

- 2026-11-09: Source inspection identifies 80 barriers,
  group-by-group scale epilogue and 6 KiB cooperative
  staging as three distinct plausible mechanisms;
  these are **hypotheses**, not measured bottlenecks.
  This was a pre-source-survey hypothesis. The hipfire
  review above changes the first-choice experiment
  to a 128×128 tile with explicitly controlled
  accumulator count; direct-vs-LDS is now secondary.
- 2026-11-09: Read hipfire native IU4 source,
  standalone A4 quantization/producer contract,
  gfx1100 measured rejection records, Strata
  local prefill kernels, hipEngine tuning rules
  and Quark→SGLang W4A16. Decisions D1–D3:
  reopen architecture research as a **bounded**
  128×128 test, not production adoption; preserve
  independent quality and exact-wire gates.

## Implementation result: independent gfx1100 128×128 / 16-wave candidate

**Experimental source:** `tools/lab/radiance/xtx27_w4a4_wide16.hip`,
never compiled into Radiance or wired to serving. The existing
`tools/lab/radiance/xtx27_w4a4.py` exposes `--kernel wide`
through the *same* tester/lease command; XDP04 `oracle` and
XDP05 `tiled` remain available. Every variant has a distinct
binary, source SHA and executable SHA manifest in scratch;
any stale binary/source combination fails closed before
a GPU test. The CLI defaults to plan/no GPU and requires
`--run` for builds or hardware. No new worktree or
third-party code copy.

### Layout / output proof

- Fixed **128 M × 128 N** workgroup, 16 wave32 /
  512 threads; `blockIdx.y` owns the M tile,
  `blockIdx.x` the N tile.
- `wave 0..15`: M subtiles `wm=wave>>1`
  (8 distinct M16 bands), N half selection
  `wn0=4*(wave&1)`. Each wave computes
  four independent N16 subtiles sequentially,
  with one int32x8 WMMA chain live at a time
  and 32 persistent f32 outputs/lane.
- Lane `l` owns `m=128*by+16*wm+(l&15)`,
  `n=128*bx+16*(wn0+c)+2*e+(l>>4)`,
  `c=0..3, e=0..7`. Card-free enumeration
  confirms **16,384 unique writers**/full
  tile; M/N partial edge stores are guarded.
- The same `XDP05` packed-tile input
  `Layout` and independent host INT32 oracle
  are deliberately copied into this private
  research binary for a clean same-data
  comparison. No change to K128 scale/zero
  correction, signed A4/unsigned W4 or
  f32 `fmaf` accumulation ordering.
- Each K128 group stages **eight A16 plus
  eight W16 subtiles**, `4096 uint32` /
  **16,384 B LDS**, using all 512 threads
  and eight words/thread. Two barriers
  per group, **80** per K5120 projection.
  This is a different workgroup arithmetic
  intensity from XDP05, not just a
  rearrangement of its 2,048 outputs.
- At M512 N17408, the grid is
  **136×4 = 544 blocks** versus
  XDP05 **272×16 = 4,352 blocks**.
  Larger tiles can reduce redundant
  global input requests but may decrease
  occupancy, especially at 512 threads.
- **Output-store risk:** native IU4 maps
  `lane&15` to M, so the unchanged
  row-major `C[M,N]` direct epilogue
  scatters stores across widely separated
  rows. Hipfire's column-major output
  favors coalesced stores. A separate
  LDS-transpose store ablation is justified
  if the direct wide kernel is still slow;
  this must preserve the *same row-major
  output* for apples-to-apples XDP05
  comparison. No store optimization
  is claimed until measured.

### Compiler findings (no GPU used)

Compiled with `/opt/rocm/bin/hipcc -O2 -std=c++17
--offload-arch=gfx1100`, extracted actual
gfx1100 code using `llvm-objdump --offloading`
and inspected `llvm-readelf -n` plus the
disassembled `v_wmma_i32_16x16x16_iu4`.
The first `#pragma unroll` eight-step K16
inner loop made a native WMMA kernel
at **235 VGPR / 65 SGPR**, 16 KiB LDS,
zero reported spills. That register
footprint is not acceptable for a
low-live-state 16-wave design.
With `#pragma unroll 1` on the eight
steps while keeping four independent
output subtiles, the same compile used
**147 VGPR / 66 SGPR**, 16 KiB LDS,
zero reported spills and zero fixed
private segment. No source code
changes to arithmetic, scaling,
output shape or memory ownership.
This is a **compiler resource result**,
not a measured latency or occupancy
measurement. The high-VGPR initial
binary and hash are preserved in
scratch as `gfx1100-w4a4-wide16-u8-prelim`
for provenance, not a promoted arm.

### Test runner / validation

```sh
python3 tools/lab/radiance/xtx27_w4a4.py --phase plan \
  --kernel wide --suite
python3 tools/lab/radiance/xtx27_w4a4.py --phase compile \
  --kernel wide --run
python3 tools/lab/radiance/xtx27_w4a4.py --phase gpu \
  --kernel wide --suite --run --device 0
python3 tools/lab/radiance/xtx27_w4a4.py --phase gpu \
  --kernel wide --shape-27b --run --device 0
```

Card-free tests include bijective
output-writer mapping, exact plan
provenance, separate manifests,
reject-invalid flags, native intrinsic
and source structure. The revised
direct/coalesced variant pair has
**56 XDP card-free tests** passing,
including XDP04/XDP05 regression
tests; Ruff passes. The hardware suite
includes 12 shapes × 2 store modes =
24 oracle cases, notably (129,129),
(128,257), (257,128) and nonmultiple-16
tails, plus full M512,N17408,K5120
sampled oracle with all outputs
nonfinite-scanned. GPU correctness
and timings are **not yet run**: RAD49
held the exclusive GPU lease at the
time of this build. Do not infer a
speedup from a successful compile.

**Decision XDP06-D4:** retain the
147-VGPR rolled K loop as the first
wide-tile GPU candidate. Reject the
235-VGPR full-unroll configuration
unless a later same-device, quality-
qualified A/B justifies the loss of
occupancy. Before considering engine
integration, test the entire small
oracle suite, full-size sampled
oracle and three independent
same-card paired benchmark
repetitions against unchanged XDP05
and pinned MXFP4; include GPU
activation quantization and BF16
model-level KL before any promotion.

## Follow-up ablation: row-major output coalescing without changing WMMA

**New mode:** `--kernel wide --wide-store
direct|coalesced`. The wide test executable
now holds **two separately compiled
template instantiations**, which share the
same device-resident A4/W4 codes and
the same native WMMA inner loop.
The default `direct` preserves the
XDP06 original direct-output kernel;
`coalesced` adds only a final
LDS transpose. A `--suite`
GPU invocation tests **both** modes
across all 12 cases with the exact
same source SHA and one exclusive lease.
A full-size run selects one mode and
records it unambiguously in its log
name and `XDP06_WIDE_PASS ... store=`
line. The Python tester still uses a
single allowed command, not a new
permission/launcher.

### Why address this separately

The GPU's native gfx11 IU4
accumulator mapping is weight-row
N in the fast accumulator lane and
activation-row M across
`lane&15`. But our pinned
`C[m*N+n]` is **row-major**.
The original direct epilogue emits
writes that stride by N rows across
neighboring lanes; eight kernel
wave outputs can write 2 adjacent N
values but not 16 consecutive M
entries in an aligned global cache
line. This may create excess global
write transactions and cause stalls.
It is a **source-level hypothesis**;
generated GPU performance counters
are needed to attribute stalls.
Changing `C` to column-major
without restoring row-major would
make the benchmark contract
non-equivalent to XDP05 and is
deliberately forbidden.

### Explicit coalesced design / safety proof

- After all 40 K128 group iterations
  and their last `__syncthreads`,
  the shared 16-KiB packed
  A/W stage can be reused.
  With transpose mode its compile-time
  allocation is `32×129 float`
  = **16,512 B**, and the
  first 4096 dwords remain identical
  to the 16,384-B packed-input layout.
- Loop four slices `s=0..3`,
  one `32 M ×128 N` output plane
  at a time. The four waves with
  `wave>>2 == s` are the sole
  plane producers. Each stores
  its 32 f32 sums/lane into
  `tile[(16*(wm−2s)+(lane&15))*129
  + 16*(wn0+c)+2*e+(lane>>4)]`.
  A *one-float row skew* (129
  instead of 128) breaks the
  pathological 16-to-1 LDS bank
  alias on column-fixed m rows.
- All 16 waves rendezvous, then all
  512 threads iterate eight
  contiguous 512-float chunks:
  `t=tid+512*i`, `i=0..7`,
  `row=t/128`, `col=t%128`,
  `out[(128*by+32*s+row)*N
  +128*bx+col]=tile[row*129+col]`
  under M,N bounds. This produces
  128 adjacent global n values
  per m row, independent of
  original WMMA lane output.
- A second barrier prevents any
  producer overwriting shared
  values while a consumer reads.
  Repeat all four slices, so
  **eight extra workgroup barriers
  per output tile**, after the 80
  group-stage barriers.
- Card-free enumerate all 4096
  LDS indices per slice and all
  16,384 global stores; **every
  location has exactly one writer
  and one reader**. No uninitialized
  edge output is used because the
  global store is guarded; padded
  output entries are initialized
  by the same zeroed 32-sum lanes.
  Host oracle/GPU pass-fail gates
  are identical to the direct arm.

### Compiled gfx1100 resources

Confirmed both actual gfx1100
`hipcc -O2` code objects,
not merely source inspection:

| Variant | VGPR | SGPR | LDS | Fixed private | VGPR spills |
|---|---:|---:|---:|---:|---:|
| direct scatter | 147 | 66 | 16,384 B | 0 B | 0 |
| LDS-transposed coalesced | 149 | 68 | 16,512 B | 0 B | 0 |

Both emit native `v_wmma_i32_16x16x16_iu4`.
The additional stage only costs
128 B LDS and 2 VGPRs in the
compiled object; whether the eight
barriers and extra LDS traffic
are repaid by global-store savings
is **unknown until GPU timings**.
Both variants meet the planned
no-spill resource screen.

### Reproducible tests

```sh
# Card-free planning + one compile:
python3 tools/lab/radiance/xtx27_w4a4.py --phase plan \
  --kernel wide --suite
python3 tools/lab/radiance/xtx27_w4a4.py --phase compile \
  --kernel wide --run

# GPU: one exclusive lease, both store modes, 12 shapes each:
python3 tools/lab/radiance/xtx27_w4a4.py --phase gpu \
  --kernel wide --suite --run --device 0

# GPU: exact full shape with independent output/sample checks:
python3 tools/lab/radiance/xtx27_w4a4.py --phase gpu \
  --kernel wide --shape-27b --wide-store direct --run --device 0
python3 tools/lab/radiance/xtx27_w4a4.py --phase gpu \
  --kernel wide --shape-27b --wide-store coalesced --run --device 0
```

**Decision XDP06-D5:** do not assume
a column-major output swap or a
store-transpose improves throughput;
run paired on the same card with
all numerical probes and identical
packed operands. If transpose wins
at M512 N17408 K5120, inspect
the produced global store
instruction/address patterns and
retain the quantization and
quality barriers to production.
If it loses, record the regression,
keep the direct store, and profile
source-copy, group folds and
effective occupancy instead.

**Status:** compiled and card-free
tested (**56** XDP tests) while
another agent holds the exclusive
GPU lease; **no GPU numerical
or performance result yet**.
No production engine patch,
model conversion or extra worktree.

## Single-lease 27B direct/coalesced paired command

The wide binary also accepts
`--shape-27b paired`; the
same tester exposes this as
`--kernel wide --shape-27b
--wide-store paired`. Within
**one exclusive agent-lease** it
runs the direct and coalesced
M512/N17408/K5120 experiments,
each with the **same deterministic
Host seed 73**, full C nonfinite
scan, 768+ CPU oracle samples
and three timed HIP event rounds.
The candidate modes have
independent run markers, and the
terminal `XDP06_WIDE_PAIRED_PASS
modes=2` marker appears only
after **both** numeric gates pass.
The Python runner additionally
requires the per-mode markers
`store=direct` and
`store=coalesced` and a
fresh source+binary SHA manifest
before accepting success.

```sh
python3 tools/lab/radiance/xtx27_w4a4.py --phase gpu \
  --kernel wide --shape-27b --wide-store paired --run --device 0
```

This is a **sequential same-lease
comparison**, not fully randomized
or clock-rotated. Because the
benchmark constructs and uploads
the deterministic Host twice,
it does not time a single shared
prepacked device allocation across
both modes. To promote a speed
claim, repeat the same-lease pair
at least three times, interleave
direction/order if controls permit,
record clocks/temperature and
compare the unchanged XDP05 and
MXFP4 kernels with their own
provenance. Workload and
numeric seeds match, but neither
synthetic quantization nor same-
shape MXFP4 constitutes a
real-model quality comparison.

**Decision XDP06-D6:** use
this single tester and one lease
for the first paired direct-vs-
coalesced 27B experiment.
Retain a nonzero exit on any
oracle or metadata mismatch;
do not derive speed conclusions
from a partial log or from the
compile-only resource screen.
The source compiles to the same
147/149-VGPR, 16,384/16,512-B
LDS no-spill code objects.
The initial 56 card-free XDP
tests pass. Hardware remains
pending while RAD49 is serving/
profiling under its GPU lease.

## Follow-up: Hipfire dataflow and per-group metadata scheduling, 2026-11-09

### Verified XDP06 wide-tile GPU result (supersedes earlier pending notes)

The first `--kernel wide --suite --run` GPU invocation ran
under an exclusive GPU lease and passed **24/24** exact-output
cases across 12 dimensions and both store modes.
`--kernel wide --shape-27b --wide-store paired --run` also
passed: **all 8,912,896 outputs** were scanned for nonfinite
values, and **768 of those** per mode were compared to
the independent int32/group128 host oracle, with **zero
reported absolute error**. At M512/N17408/K5120, three
HIP-event rounds within the same leased run produced:

| XDP06 128×128 / 16 wave | Minimum | Median | Maximum |
|---|---:|---:|---:|
| Direct row-major stores | 2,267 µs | **2,311 µs** | 2,317 µs |
| LDS-transposed, coalesced stores | 1,933 µs | **1,952 µs** | 2,019 µs |

This is **15.5% faster** for the same wide-tile kernel after
coalescing, but **~11.7% slower** than XDP05's previous
1,747-µs, 32×64 kernel and **~2.12× slower** than the
previous ~923-µs pinned MXFP4 control. They are not a
fully clock-rotated three-run A/B, and none includes GPU
A4 quantization or original BF16 quality measurement.
Logs: `${BC_WORK_ROOT}/xtx27-prefill/w4a4/gpu-wide16-suite.log`
and `gpu-wide16-27b-paired.log`.
A later XDP05 rerun was rejected by the exclusive lease
because RAD49 resumed GPU work; **no value from that
rejected rerun is used as timing evidence**.

This observation invalidates the simple hypothesis that a
larger 128×128 tile alone explains Hipfire's better timings.
The output layout also matters and a more efficient
metadata/codegen pipeline is needed.

### Hipfire source-level comparison: where the remaining gap is

Reviewed the default-branch hipfire
`kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip`
and its accompanying historical gfx1100 Lloyd-split
investigation. Its native IU4 kernel stages packed weight
data **plus dual f16 weight scale/zero headers** and per
128-element activation sidecars `(f32 scale, i32 sum,
64 packed bytes)` into LDS. The consumer reads its group
headers from LDS at the point of integer-accumulator fold.
Its tuned 16-wave variant has 32 f32 sums/lane, one live
int32 chain, and a compiler scheduling fence between
subtiles; hipfire notes a **98-VGPR** low-live-state
configuration in its own compiler experiment. These
details belong to hipfire's MQ4G256V2 tensor encoding and
column-major output: the model format and quantization
scheme are **not** interchangeable with BigCherry's
current row-major W4A4 research representation.

By contrast, before this follow-up BigCherry's wide
kernel read `sa[m,g]`, `sw[n,g]`,
`asum[m,g]` and `wz[n,g]` from global memory
**inside each output fold**, with 40 K128 groups and 32
f32 persistent sums/lane. Even if L2 caches these values,
the generated code maintains additional addresses,
scalar-to-vector flows and live intermediates. The
compiler reported 147/149 VGPR per direct/coalesced
thread. Hypothesis: *metadata address/lifetime, not
simply WMMA arithmetic or 80 barriers*, is a large
part of this gap. GPU timing is still required.

### Controlled, independent code experiments

The existing 128×128 / 16-wave kernel now has **two
optional compile-time code ablations** (both off in the
XDP06 default `--kernel wide`):

- **`wide16sched`**, wrapper
  `xtx27_w4a4_wide16_sched.hip`, invokes
  `__builtin_amdgcn_sched_barrier(0)` after each
  16×16 fold. The compiler generated **exactly the
  same resource allocation**, direct 147 VGPR /
  66 SGPR /16,384 B LDS and coalesced 149 VGPR /
  68 SGPR /16,512 B LDS; no spills. Therefore
  **scheduling fence alone is not the solution**,
  and the arm is deprioritized pending a specific
  measured reason to benchmark it.
- **`wide16meta`**, wrapper
  `xtx27_w4a4_wide16_meta.hip`, stages four
  128-element header arrays per K128 group:
  A f32 scale, A exact int32 sum, W f32 scale,
  W int32 zero. Four nonoverlapping 128-thread
  cohorts perform 512 total header loads, all
  before the existing first group barrier.
  The same signed-A4×unsigned-W4 8-WMMA chain,
  i32 subtraction and f32 per-group
  multiplication/fmaf remain unchanged.
  The compiler produced a substantial
  **register-allocation reduction**:

  | Variant | VGPR | SGPR | LDS | Spills | Fixed private segment |
  |---|---:|---:|---:|---:|---:|
  | wide direct (unchanged) | 147 | 66 | 16,384 B | 0 | 0 |
  | wide coalesced (unchanged) | 149 | 68 | 16,512 B | 0 | 0 |
  | wide16sched direct / coalesced | 147 / 149 | 66 / 68 | unchanged | 0 | 0 |
  | **wide16meta direct** | **104** | 68 | 18,432 B | 0 | 0 |
  | **wide16meta coalesced** | **110** | 70 | 18,560 B | 0 | 0 |

  These are actual `hipcc -O2 --offload-arch=gfx1100`
  and `llvm-readelf -n` results, not an assumed
  occupancy improvement. Both variants still emit
  real `v_wmma_i32_16x16x16_iu4`, 0 reported
  spills and 0 fixed private segment.

- **96-VGPR cap is rejected**: a scratch-only
  `__launch_bounds__(512,2)` +
  `amdgpu_num_vgpr(96)` build of staged metadata
  forced 8 VGPR spills/36 B private per direct
  thread, and 12 spills/52 B private per coalesced
  thread. Do **not** force an attractive 96-VGPR
  metric by introducing new global scratch traffic.
  This failed compiler preflight is retained
  under `${BC_WORK_ROOT}/xtx27-prefill/w4a4/meta96-preflight*`;
  it is **not** a selectable GPU arm.

- **`wide8`** is the separate 8-wave/256-thread
  128×128 candidate from the prior ablation.
  Its 64 f32 sums/lane resulted in **165 VGPR**
  (direct and coalesced), 91/93 SGPR and
  16,384/16,512 B LDS, no spills. That
  higher register count than the 16-wave
  staged metadata option means it is a
  valuable experimental control but not
  an occupancy win by construction.

All source variants are BigCherry-owned, isolated
standalone HIP programs. No third-party kernel was
copied or executed and no vendor files were modified.

### CLI, provenance and GPU acceptance

The existing `xtx27_w4a4.py` command
supports the `oracle`, `tiled`, `wide`,
`wide8`, `wide16sched` and `wide16meta`
kernels, each with a **distinct scratch binary,
manifest and log prefix**. The two new
include-based wrappers have a composite SHA-256
over the wrapper and the included base source,
so a modification to either invalidates a
compiled binary. `--phase plan` never builds
or acquires a GPU, and `--phase compile --run`
never launches GPU kernels. `--phase gpu --run`
requires a fresh matching source/binary SHA
and acquires an exclusive `agent-lease`.

The GPU tests are **pending** for `wide8`,
`wide16sched`, and `wide16meta`. The
highest-priority new arm is `wide16meta`;
there is no speed or model-quality claim from
its successful compile. When GPUs are free:

```sh
python3 tools/lab/radiance/xtx27_w4a4.py --phase gpu \
  --kernel wide16meta --suite --run --device 0
python3 tools/lab/radiance/xtx27_w4a4.py --phase gpu \
  --kernel wide16meta --shape-27b --wide-store paired --run --device 0
python3 tools/lab/radiance/xtx27_w4a4.py --phase gpu \
  --kernel wide8 --suite --run --device 0
python3 tools/lab/radiance/xtx27_w4a4.py --phase gpu \
  --kernel wide8 --shape-27b --wide-store paired --run --device 0
```

First require **24/24 full-output small GPU oracle
cases per kernel** and then full-size
nonfinite scan + **768 sampled integer oracle
checks** per mode. Only after correctness, compare
paired full-shape medians with `wide`, XDP05
and pinned MXFP4 on the **same GPU**.
Repeat and rotate arms when the lease permits,
tracking GPU clocks/conditions. Any new
GPU A4 quantization, tile-major weight
repacking or quantizer quality cost remains
excluded from this isolated kernel timing
and must be included before runtime promotion.

**Decision XDP06-D7:** staged metadata is the
leading path because it is a measured
compiler-resource improvement and more
closely matches the benchmarked hipfire
dataflow. It is not yet a measured
performance improvement. Reject the
96-VGPR forced-spill build, deprioritize
the no-impact scheduling fence and retain
the 8-wave case as a controlled topology
comparison. Do not modify the production
engine or convert the 27B checkpoint
until isolated performance and real-model
KL pass.

## Measured continuation and decisions

See [XDP07](XDP07.md) for the completed same-XTX 24-case GPU oracles, three independent staged-header 27B projection runs, two group-major-header runs, the 8-wave rejection, source/ISA allocation evidence and quality/promotion gates. The 16-wave staged-header/coalesced median reached **1,056.9 us**, versus fresh XDP05 **1,762.5 us** and MXFP4 **927.9 us**. XDP06 and XDP07 remain experimental; no production kernel or BF16 model conversion was changed.
