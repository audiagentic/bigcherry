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
