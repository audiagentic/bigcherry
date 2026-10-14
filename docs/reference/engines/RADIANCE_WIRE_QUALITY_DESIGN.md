# Radiance tensor-parallel wire-quality redesign

**Scope:** BigCherry libr3 gfx1100 and the pinned Radiance 1.3.0 (89cee7ce414486c717aa2085c99e1f522e4c0ea6) two-rank no-P2P transport on two RX 7900 XTX cards, first Qwen3.8-Flash-Next then Qwen3.8-27B. Work plans RWQ01–RWQ06 at docs/planning/active/patching-radiance-wire-quality/.

## Evidence and decision

Flash-Next uses native .rad with 4-bit experts, INT8 trunk; its quantisation quality versus a BF16 teacher is a separate topic. Here 'wire KL' means comparing the same .rad on the same engine with exact two-card communication versus the same weights under compressed transport. On 42,761 scored positions, WHT6 relative to exact on Flash-Next is mean KL 0.0746 and 87.18% top-1 (PPL 10.436 versus 10.106), while production exact is deterministic. At 7.2K prompt length, measured prefill exact 1263 tok/s and WHT6 1343 tok/s: modest ~6% gain for substantial quality loss. On 27B, WHT6 adds about 0.000757 mean KL over exact versus a third reference (0.004422 vs 0.003665), only ~0.4 percentage points of top-1, for about +38% prefill, so do NOT automatically replace its current WHT6 production choice.

### Baseline and source links

- BigCherry measurement: engines/radiance/README.md, sections KL, cross-card timing, Flash-Next on two XTX and exact-vs-WHT6.
- Pinned source: https://github.com/Quackster/radiance/blob/89cee7ce414486c717aa2085c99e1f522e4c0ea6/libr4d/r4d_ar_oneshot_2rank_wht6.h ; .hip companion; libr4d/r4d_ar_entry.hip.
- Graph declaration: arch/common/rad_block_hc.h; arch/common/rad_block_moe_fp8.h; arch/common/rad_arch.h; arch/qwen35_fp8/qwen35_fp8.cpp.
- BigCherry component: engines/radiance/components/libr3/native/r4d_ar_entry.hip.rw, component.toml; engines/radiance/recipes.toml and docs/reference/engines/RADIANCE_RUNBOOK.md.
- Existing experimental R3_AR_WIRE4 on 27B: +5.6% prefill but mean KL 0.0044 -> 0.0126; do not promote without independent policy/quality proof.
- Existing R3_AR_PIPE tried eight slices within one GPU kernel: 2.0–2.15 ms -> 2.62–2.74 ms for a large reduction, prefill slower by ~6%; do not repeat without a different handshake/launch structure.

## Data path and protocol invariants

The two ranks write/read mapped pinned host scratch with no P2P. Device-side release/acquire system-scope flags track epoch; local pack must reflect precisely the bytes sent to peer; GPU graph replay and two-slot parity must be safe under out-of-order scheduling. The pinned WHT6 code rotates each contiguous group of 64 BF16 inputs through a normed Walsh-Hadamard; calculates amax; quantises to signed 6 bits with BF16-rounded group scale; writes codes and BF16 peer scale; retains packed local codes and f32 local scales; after the peer flags become ready, adds dequantised local/peer rotated rows in fp32; inverse-rotates and executes post-ops. WHT groups are independent. Ensure no FMA contraction causes noncommutative per-rank output; keep the pinned compiler condition -ffp-contract=off where required. The WHT6 push/read ordering and thread-safe shared scratch are a correctness protocol, not merely compression implementation detail.

For one 64-element group with one BF16 scale: exact BF16 = 128 bytes, WHT4 = 34, WHT6 = 50, WHT8 = 66. Equivalent effective bits/value: 16, 4.25, 6.25, 8.25. Ratios to exact: 100%, 26.5625%, 39.0625%, 51.5625%. This is transmitted per rank, excluding flags, padding and launch overhead. WHT8 CHUNK_GROUPS=32: 2048 values/chunk, 2048 byte codes per chunk, 32 BF16 scales (64 B). WHT8 logical payload is 2112 B/chunk including scales, though actual physical code and scale planes are separated. WHT6 corresponding 1536+64=1600 B/chunk. For a 2048×2560 BF16 activation: 10,485,760 B exact, 4,096,000 B WHT6 and 5,406,720 B WHT8. These figures are transport payload projections, not measurements of PCIe transactions.

The current large no-P2P reductions on 27B are link-bound: the 7.17MB WHT6 message took about 2.0–2.15ms median, exact 18.35MB ~5.0ms, with two opposite host-link crossings. However Flash-Next prefill bottleneck is expert movement and GEMMs as well as collectives; reduced payload alone is insufficient to predict total benefit.

## Kernel development sequence

1. RWQ01: instrument *issued*, not merely declared, all-reduces. Include fused ar_rmsnorm_quant_fp8, ar_hc_write and ar_gather_hc_write. Ensure standalone all_reduce and disjoint embedding summation remain exact as currently intended. Freeze reference build/model/corpus and control for native kernels, KV, draft, helper.
2. RWQ02: WHT8 as named experimental substitution behind WHT6 transport only for initial A/B, on both ranks, default off, activated in log. Use qmax=127 qmin=-128; scale=bf16_round(amax/127), RNE quantisation, same group and inverse. Preserve scale plane at slot_bytes/2 and prove locpack and scratch capacity for max graph. The transition to independent wire=8 op declarations, if needed, must go through a pinned Radiance patch package; do not ship an opaque wire=6 masquerade as a production feature.
3. RWQ03: add an immutable model-aware transport policy with exact as default. Key by semantic operation family, phase, layer, and M. Only choose from declared supported operators, taking graph contract and fused capability into account. All ranks resolve the same policy hash before any handshake. Add family-level exact-vs-lossy ablations before per-layer search. A protected operation that lacks exact fused support must fall back through a correct explicit reduce + epilogue, with measured performance cost. Do not mask a missing reduction.
4. RWQ04: only if real prefill activation histograms justify it, test one residual correction per WHT6 group, explicit 53 bytes/64 logical (before padding); or controlled clipping/scale choices. Receiver folds residual in rotated space before inverse, recomputing neither local bytes nor scale. Compare against eight-bit at matched wire bytes.
5. RWQ05: run 42,761-position model KL and held-out code/tool/long-context, and ABBA performance at multiple depths. Gate on mean KL <=0.005, top-1 >=99%, PPL within ±0.5%, p99 <=0.05 with p99.9 tail review and >=3% repeatable prefill improvement over exact. These are *proposed* adoption targets; final thresholds require explicit owner approval, and the exact fallback wins if none pass.
6. RWQ06: only promote measured winners; profiles describe non-default settings only, quality observations use existing engine_quality/ab-benchmark, and pinned source/package tests protect engine changes.

## Selection and numerical hazards

- Tiny decode and MTP verification are likely better exact; size floor already makes many compressed paths fall back, but the fused HC paths have their own wire declared at graph build. Do not assume one global size check catches all.
- The model's embedding all_reduce sums disjoint nonzero rows; any lossy treatment there is pure damage. Preserve exact.
- Different layer/operation paths can have drastically different sensitivity; do not use input-dependent rank-local packet decisions that can diverge and deadlock.
- BF16 rounded scale can under-represent amax; clamp must be tested on qmin/qmax, all zero and very large/small finite values. Non-finite activations must have a defined error or exact fallback, not garbage symbols.
- The existing paired exact and WHT6 fused graph handles have different declared row limits. Do not substitute operator IDs without verifying supports_m and fused epilogue. Count fallback work and timing.
- A low tensor rel-L2 is not proof of logits KL. Testing only generated text misses tail errors; testing only one corpus overfits the policy.
- Synthetic proxy's logical bit packing is **NOT the GPU layout**, and Python float32 emulation is not a HIP oracle; only the GPU device checker can establish physical memory and post-op correctness.
- Measured speed must be unprofiled. R3_AR_TIMING synchronises the rank stream before and after each collective, altering throughput.

## Test ladder and evidence

T0 offline: WHT/inverse, RNE BF16, packet pack/unpack, rank swap, tails, zero, synthetic normal/outlier/sparse samples (RWQ01 proxy and seven unittests). T1 device: standalone and all fused operations, guard pages, odd/valid shape boundary, graph replay and system-flag epoch; both GPUs compare to same host oracle. T2 model: exact rerun determinism then full per-position KL and held-out sections. T3 speed: ABBA >=3 paired repetitions each arm, unprofiled prefill/decode, mapped-host bytes and kernel timings separately, fixed topology/model build. T4 deployment: mismatch policy rejection, default/profile validation, all supported paths with helper and drafter independently controlled.

Every acceptance report declares {engine pin, source tree hash, libr3 build hash, model hash, patch IDs, profile, corpus hash, GPU topology, ROCm revision, environment switches, precision policy hash, timestamp, repetition count, KLD direction, p50/p99/p99.9/top1/PPL, prefill/decode and MTP acceptance}. Tracked compact run summaries belong under docs/evidence/RWQxx; raw machine traces under artifacts/RWQxx-<run> as prescribed by tooling conventions.

## Existing offline mock (no engine mutation)

Run python3 tools/lab/radiance/wire-quality/codec_mock.py --json and python3 -m unittest discover -s tools/tests/radiance -p test_wire_quality_mock.py -v. Seven tests passed locally on 2026-10-15. docs/evidence/RWQ01/mock-results.json is synthetic; it is not a KLD measurement and is not promotion evidence. Hardware tests are not possible in this session; every GPU claim remains pending.
