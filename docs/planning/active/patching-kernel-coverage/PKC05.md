---
id: PKC05
order: 0
plan: patching-kernel-coverage
state: pending
created-at: '2026-10-01T05:39:29.192524+00:00'
breadth: ''
skill: advanced
created-by: agent
work: L
---

# Quantized MMVQ/MMQ headroom: Q8_0 geometry, Q2_K register pressure, Q1_0 AMD unpack

## Description

Own quantized CUDA/HIP MMVQ/MMQ kernel headroom as one capability instead of creating per-quant plan items. Existing Q8_0 work covers decode/prefill attribution, loader geometry and MMQ tuning. Upstream PR #29910 supplies a Q2_K register-pressure mechanism; PR #29927 supplies an AMD-native Q1_0 byte-permute/unpack mechanism. Both remain qualification candidates against the current BigCherry pin rather than separate plans.

Do not create Q1_0, byte-permute, Q2_K, generic VGPR, or generic AMD-intrinsic plans. These mechanisms share PKC05's correctness, compiler-resource, kernel-timing and end-to-end evidence pipeline. Architecture/path selection remains owned by existing HIP-autotune machinery.

27B Q8_0 plain decode is ~36 t/s (38.4 with adaptive AllReduce) against a ~66 t/s DRAM roof. Q8 remains the representative production attribution lane; Q2_K and Q1_0 are targeted mechanism probes whose promotion requires local RDNA evidence.

## Fresh upstream evidence

### Q2_K: PR #29910

The patch reduces aggressive unrolling/removes a temporary loop. Reported spill elimination on AMD shows compiler pressure can dominate some MMQ shapes, but reported throughput varies strongly by ubatch. Zero spills are diagnostic evidence, not a throughput gate.

### Q1_0: PR #29927

The candidate replaces CUDA-compatible `__byte_perm` emulation with AMD-native `__builtin_amdgcn_perm` in the Q1_0 decode dot-product and MMQ tile unpack paths under HIP. Published gains are from older AMD hardware and are motivation only; qualify the exact upstream change on gfx1100/gfx1201 and inspect generated ISA before considering broader intrinsic substitution.

Against current BigCherry pin `050439614`, #29910 and #29927 are still external candidates and must be tested by pinned PR SHA or equivalent local patch, not assumed present.

## Steps

1. Q8 attribution: rocprofv3 per-device critical-path time per category/token using `tools/lab/profiling/profile-27b-q8.sh`.
2. Q8 loader: qualify cooperative aligned Q8_0/F32 MMVQ loader only if attribution still supports it.
3. Q8 geometry/MMQ: sweep gfx1100 MMVQ 4x1/8x1/4x2/8x2 and measured MMQ tile candidates; promote only winners.
4. Q2 exact-upstream gate: qualify #29910 on gfx1100 + gfx1201 with `MUL_MAT`/`MUL_MAT_ID` correctness, VGPR/spills, rocprof kernel timing and pp2048 ub16-512.
5. Q2 dispatch only if required: if local architectures show a crossover, compare unroll 1/2/4 through existing HIP autotune. No second dispatch registry.
6. Q1 exact-upstream gate: reproduce #29927 unchanged. Build native gfx1100/gfx1201 variants and a control using the current compatibility path. Cover both MMVQ/decode and MMQ/prefill unpack sites.
7. Q1 compiler/ISA gate: retain compiler outputs/disassembly. Confirm the intended AMD permute lowering without extra lane-shuffle, pack/unpack, VGPR, spill or occupancy regressions. ISA is explanatory evidence, not a promotion criterion by itself.
8. Q1 performance gate: backend-op microbench n=1/2/4/8/32/128/512 plus pp512/2048/4096 and tg128/512; MMQ ub16/32/64/128/256/512; identical model/build/power settings and repeated samples.
9. Cross-quant consolidation: store Q8/Q2/Q1 evidence in one schema and use compiler/ISA evidence to prune candidates before expensive campaigns.
10. Broader intrinsic audit only after Q1 success; any follow-up stays in PKC05 unless it is a genuinely different subsystem.

## Detailed Solution & Technical Design

PKC05 treats three optimization classes as one kernel-coverage problem:

- Q8_0: memory transaction/geometry efficiency;
- Q2_K: register pressure and source-level unroll effects;
- Q1_0: architecture-native packed-byte permutation/unpack efficiency.

The common gate is `correctness -> compiler/ISA resources -> kernel timing -> repeated end-to-end`. Compiler counters explain results but never substitute for throughput.

For Q1_0, preserve CUDA behavior and scope native AMD builtin use strictly to HIP. Do not create a generic wrapper unless multiple call sites/formats prove a measurable need. Verify selector semantics with backend-op vectors; matching signatures are not proof of identical byte selection.

For Q2_K, reject spilling variants when a zero-spill equivalent exists, but still require end-to-end confirmation because lower VGPR pressure can trade against ILP.

## Stream-K follow-up: upstream PR #30022

PR #30022 is a draft GCN tuning series, not RDNA evidence. Its useful mechanism is narrower than its title: `launch_mul_mat_q()` computes regular-tile efficiency, uses `nsm * config.occupancy` Stream-K blocks for GCN, bypasses Stream-K when tiled efficiency is >=90% or there is enough tiled work, and restores full wave64 fixup parallelism instead of the NVIDIA-oriented half-sized fixup. It also flips many entries in `mmq-config-gcn.cuh` to Stream-K and retunes selected tile widths.

Do **not** port the GCN config table or thresholds to gfx1100/gfx1201. The PR explicitly gates the new policy with `GGML_CUDA_CC_IS_GCN(cc)`, leaves CDNA as TODO, has no benchmark data yet, and states that Q1_0/Q2_K retesting waits on #29927/#29910 plus #30021. For BigCherry this is therefore a mechanism probe only: if PKC05 profiling shows MMQ shapes with low regular-tile efficiency or fixup overhead on RDNA, reuse the existing HIP-autotune lane to compare native tiled versus current Stream-K with the same shape; otherwise terminate this branch of work without adding dispatch state.

Cheap discriminator before hardware: extend the existing evidence extraction to record `ntiles_dst`, CU count, selected MMQ occupancy, regular-tile efficiency and whether Stream-K/fixup launched for representative Q8/Q2/Q1 shapes. Only schedule an RDNA A/B if Stream-K is selected while regular tiling is >=90% efficient, or fixup time is >=3% of MMQ kernel wall time. Promotion requires >=3% repeated kernel-time improvement and >=1% end-to-end improvement with no >1% regression across primary pp/tg controls. Failure closes the Stream-K sub-slice; it does not create a new scheduler or dispatch table.

## Evidence schema

```text
{arch, rocm, compiler, quant, path, ubatch_or_n, geometry, intrinsic, unroll,
 vgpr, sgpr, spill_store_bytes, spill_load_bytes, occupancy,
 kernel_ms, pp_tps, tg_tps, correctness, isa_fingerprint}
```

Pin upstream base SHA and candidate PR SHA so evidence remains reproducible if a PR moves.

## Files

- `vendor/llama.cpp/ggml/src/ggml-cuda/mmvq.cu`
- MMQ implementation/config including `mmq-config-rdna3.cuh`
- Q1_0 quant/dot helper containing its Q8_1 dot path
- Q1_0 MMQ loader/unpack path
- `tools/lab/profiling/profile-27b-q8.sh`
- existing HIP-autotune/dispatch and evidence tooling
- temporary BigCherry patch only while the upstream candidate is absent from the pin

## Validation

Q8: existing tierL-qwen27b-q8 experiment contract, repeated paired sessions, positive improvement CI and controls within noise.

Q2: backend-op ROCm `MUL_MAT` + `MUL_MAT_ID` with q2_K; gfx1100/gfx1201 compiler resources; pp2048 ub16-512 repeated samples; no >1% regression in primary larger-ubatch lanes if specialization promotes.

Q1 correctness: packed patterns around byte/word boundaries; MMVQ and MMQ both exercised; gfx1100/gfx1201 required, gfx1030 useful control; existing quantized tolerances only.

Q1 performance: n=1/2/4/8/32/128/512 microbench, pp512/2048/4096, tg128/512, MMQ ub16-512; record kernel time, end-to-end t/s, VGPR/SGPR/spills, occupancy and instruction count where available. Require repeated median + dispersion.

## Consolidation / duplication audit

PKC05 owns quantized MMVQ/MMQ kernel headroom and architecture-native unpack experiments. `patching-hip-autotune` owns dispatch policy. MET02 owns range-aware MoE semantics and may consume improved kernels but must not duplicate them. If #29910/#29927 merge before qualification, compare pinned pre/post commits and retain only BigCherry evidence or architecture-specific dispatch work not supplied upstream.

## Effort & Risk

Q1 direct qualification is low implementation risk but high extrapolation risk. Q2 is low implementation/medium tuning risk. Avoid broad intrinsic rewrites until exact upstream candidates are measured on gfx1100/gfx1201.

## Acceptance Criteria

- Q8 attribution still identifies actionable MMVQ/MMQ headroom before custom Q8 changes promote.
- #29910 or merged equivalent is qualified on gfx1100/gfx1201 with correctness, compiler resources, kernel timing and end-to-end evidence.
- #29927 is qualified on gfx1100/gfx1201 across both Q1_0 MMVQ and MMQ unpack sites.
- Q1 generated ISA is captured and correlated with performance; no claim relies on older-architecture results alone.
- No duplicate quant/intrinsic/dispatch plan is created.
- Any promoted candidate has positive workload-relevant end-to-end evidence and no material control regression.

## References

- https://github.com/ggml-org/llama.cpp/pull/29910
- https://github.com/ggml-org/llama.cpp/pull/29927
- https://github.com/ggml-org/llama.cpp/pull/28398
- https://github.com/ggml-org/llama.cpp/pull/30022

## Notes

This absorbs the strongest useful side-branch quant-kernel mechanisms into the existing kernel-coverage owner rather than creating Q1_0/Q2_K child plans.

## Change Log

- 2026-10-01T05:39:29.192524+00:00 (created-by): Created by agent
- 2026-10-05: Transplanted Q2_K register-pressure and Q1_0 AMD-native unpack qualification from `automation-qfp-indexer-20261004`; aligned upstream status to current pin `050439614` and preserved single-owner dispatch policy.
- 2026-10-07: Classified draft upstream #30022 as GCN-only evidence; added bounded RDNA Stream-K/fixup discriminator under existing PKC05/HIP-autotune ownership.
