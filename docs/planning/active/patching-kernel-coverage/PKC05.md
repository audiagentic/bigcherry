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

## 2026-10-13 Q1_0 HIP unpack discriminator (BCOP188; current pin b11552)

**Scope and evidence:** This subsection supersedes only the historical Q1_0 qualification wording above, not the Q2_K or Stream-K ownership. Current `config/recipes.toml` pins llama.cpp **b11552** (commit `23b0202a189c44a54625aadcb37a946dd1d6278d`), not the older `050439614` cited above. Upstream [#29927](https://github.com/ggml-org/llama.cpp/pull/29927) is **open, non-draft**, head `b6c4ff7c6e36c92f9a5849f848c7b7a03a6a645c` (last updated 2026-10-09). The pin and current upstream still use the old unpack. These are source observations, **not** an AMD benchmark or evidence of activation in a BigCherry model.

**Exact implementation and call paths:**
- `ggml/src/ggml-cuda/vecdotq.cuh::vec_dot_q1_0_q8_1` consumes two signed `int16_t` packed words per 32-weight Q8_1 chunk, performs ten `__byte_perm` calls per word, then four `ggml_cuda_dp4a` accumulations per word; `mmvq.cu::get_vec_dot_q_cuda` selects it for Q1_0. The scale multiply stays outside the unpack. The candidate replaces the 10-call chain with four `__builtin_amdgcn_perm` calls per word **only under `GGML_USE_HIP`**, leaving the non-HIP path intact.
- `ggml/src/ggml-cuda/mmq-load-tiles.cuh::ggml_cuda_mmq_load_tiles_q1_0` repeats the same 16-bit unpack for shared-memory tiles. `mmq.cuh::ggml_cuda_mmq_get_util_funcs` selects this loader for both DP4A and MMA layouts; do not qualify decode alone and claim prefill coverage. `mmq.cu::ggml_cuda_should_use_mmq` admits Q1_0 subject to the existing >=48 KiB shared-memory check; `ggml-cuda.cu::ggml_cuda_mul_mat` tries MMVQ before MMQ. Record the **executed** route, physical shapes, GPU and graph state, not just the model quant name.
- For each 16-bit word `q`, each nibble `g=0..3` encodes four signs. #29927 uses `bits=(uint16_t)q >> (4*g)`, `idx=(bits&1)|((bits&2)<<7)|((bits&4)<<14)|((bits&8)<<21)`, and `__builtin_amdgcn_perm(0x000001FF,0x000001FF,idx)` to emit four signed bytes (-1/+1). The cast to unsigned 16-bit is material to signed-promotion boundary tests. No changes to quant storage, Q8_1 activations, dot-product order, scales, scheduler, allocations, transfers, dispatch tables or graph lifetime are authorized.
- Older [#28398](https://github.com/ggml-org/llama.cpp/pull/28398) is an **alternative** native-permute decode-only approach (arithmetic bit spread + one perm per nibble), not a prerequisite or second patch to stack. #29927 covers both MMVQ and MMQ. Compare #28398 only if #29927 fails compiler-resource/performance qualification; never combine the two.

**Cheapest discriminator completed, not hardware qualification:** A disposable Python host fixture reproduced the pinned `__byte_perm` selection semantics, #29927's AMD byte-selector mapping, #28398's 0x0C/0x0D constant mapping and an independent scalar -1/+1 oracle. **65,536/65,536** signed-16-bit patterns matched all four output words across all three implementations; **4,096/4,096** deterministic signed-int8 dot products and **6/6** boundary patterns matched. Source checks: **12/12** (two HIP-gated sites, four native permutes each, unchanged CUDA branch, Q1 MMQ admission, etc.). This proves the *host-model mapping only*: no HIP compilation, generated ISA, GPU memory safety, backend-op run or throughput was measured. A reduction from 20 source-level `__byte_perm` calls to 8 native-permute calls per 32 sign weights is **not** an instruction-count or speedup measurement.

**Model and ownership fence:** `config/models.toml` currently registers **six** contract models, none Q1_0; this does not prove a Q1_0 GGUF is absent from Brutus storage. Do not queue model benchmarks, add a guessed registry entry or infer production gain until an actual Q1_0 model is verified by path, quant metadata, size and hash. PKC05 owns Q1_0 kernel semantics; existing HIP autotune owns architecture/shape policy; QFP44 owns generic MMQ tile geometry; PRBE111/1273 owns IQ (not Q1_0) MMVQ tuning. No second Q1 plan, selector, kernel registry or telemetry layer.

**Bounded next gate and terminal decisions:**
1. Freeze control b11552 and subject b11552 + exact #29927 head; recheck upstream before applying. First run existing patch-application/composition checks with serving-core and the 1273 vecdotq edits, without touching protected Radiance/1340 or modifying the rejected 1000 Q2_K patch. Build HIP separately for gfx1100 and gfx1201 (gfx1030 optional); verify `v_perm_b32` lowering, VGPR/SGPR, scratch/spills, register occupancy, code size, launch geometry and that both Q1 call sites are compiled. CUDA build/control must remain unchanged.
2. Run existing `test-backend-ops` Q1_0 `MUL_MAT` and `MUL_MAT_ID` across n=1/2/4/8/16/32/64/128/512, irregular/tail rows, single and multi-ubatch, and DP4A/MMA where dispatch permits. Compare F32 outputs to stock/CPU oracle and preserve full logits, greedy/MTP acceptance and graph replay **when an actual eligible model exists**. Reject on mismatch, invalid read, missed Q1 path or compiler regression. No hidden work removal.
3. Only with an eligible, hash-bound Q1_0 model and executed Q1 kernels, profile exposed MMVQ/MMQ kernel time and end-to-end wall time by prefill versus decode. If eligible kernel exposure is below **2.9126%** of end-to-end latency, even eliminating all of it cannot yield a 3% throughput gain: close `INSUFFICIENT_HEADROOM` without a hardware A/B. If no model is registered, stop at `AWAIT_Q1_MODEL`; do not call that a failed performance experiment.
4. For surviving lanes, compare stock vs exact #29927 in >=4 independent fresh-process sessions/architecture, >=10 paired ABBA rounds, pp512/2048 and tg128/512, matched model/context/clock/driver/ROCm/power. Require >=3% end-to-end improvement with positive CI95 lower bound, <=1% unrelated-quant/control regression, unchanged correctness and no graph/memory regression. If the upstream PR merges before the next pin, adopt upstream and keep only BigCherry qualification evidence. Otherwise reject/park; never promote from MI50 or gfx1201 upstream numbers alone.

**External comparison:** [vLLM GGUF plugin](https://github.com/vllm-project/vllm-gguf-plugin) publishes Q4/Q6/IQ4/Q8 reference lanes but no Q1_0 RDNA qualification; [SGLang](https://github.com/sgl-project/sglang) provides no verified transferable Q1_0 native-permute mechanism here. #28398 reports a single-gfx1201 Q1_0 decode result using a *different* implementation; it is external evidence only, not a BigCherry measurement. Current dual-XTX/R9700 no-P2P transport and gfx1030 chipset placement do not affect this single-device byte-unpack hypothesis; cross-device claims require separate evidence.

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
