---
id: PEF07
order: 7
plan: patching-external-fixes
state: pending
created-at: '2026-10-04T00:30:00+00:00'
breadth: cross-plan
skill: advanced
created-by: agent
priority: P0
work: L
---

# Retained decode submission + overfit-engine mechanism qualification

## Description

External-engine scan found one genuinely new BigCherry seam and several useful mechanisms that already have owners. The new seam is **retained decode submission**: eliminate recurring HIP host launch overhead after QFP13 has reduced/fused individual graph nodes. Hipfire demonstrates the mechanism directly on Radeon AI PRO R9700/Qwen3.6: a stable decode tape contains 833 launches; retained PM4 replay is recorded as one vendor AQL packet and improves 164.220 -> 174.087 tok/s (~+6.0%) over tuned HipGraph. A retained-PM4 arm plus removal of one proven-independent wait reaches 178.320 vs 165.839 tok/s (~+7.5% combined). This is transport/submission optimization, not another kernel fusion, graph-cap, or AllReduce redesign.

Other engines are evidence sources, not new patch families:
- **NInfer** prepares model-specific CUDA graphs and reserves DFlash/KV address-space allocations before capture. Import the invariant: graph-visible addresses/resources are prepared before capture; do not revive QFP06's disproven fixed graph-cache cap.
- **Gufo** runs a batch shape eagerly before HIP graph capture so its GEMM arena can grow outside capture. Import warm-then-freeze allocation discipline; avoid decode-path heap growth.
- **Splash** combines specialized exact-shape kernels, DFlash2, automatic memory planning, prefix reuse and adaptive batching. Import shape-specialized dispatch, bounded memory planning and lifetime-aware staging; do not port Metal code.
- **DwarfStar/ds4** keeps its resident SSD expert cache warm across sessions and has optional expert look-ahead/prefetch machinery. Residency/prefetch policy belongs to MET, not a second expert-cache subsystem.
- **llamAmpere** specializes speculative verification widths: its ternary verify path unpacks a weight block once per lane and dots multiple verification columns, instead of re-unpacking per column; its attention path separately specializes verify widths 6-8. Import multi-column reuse/verify-width specialization into the existing MTP/MMVQ/FA owners, not SM86 CUDA kernels.
- **Strata** (interpreting the requested "rata" as the active MoE inference-engine family; `ro99/strata` is the verified C++/CUDA project) makes admission/placement and bottleneck attribution first-class. Its own measurements explicitly reject faster compute kernels when expert staging/PCIe dominates: streaming MoE cases spend only ~2-6% in GPU matmul and show ~1.0-1.06x ceilings. Import that *ceiling gate* into MET/kernel promotion: do not optimize a compute kernel whose measured wall-time share cannot pay back the added code.

## Steps

1. Stage 0: from QFP13 production traces, establish decode topology stability on gfx1100 and gfx1201. Hash the ordered launch sequence plus kernel identity, grid/block/shared-memory shape, dependency/fence points, graph-shape key and pointer-generation IDs. Report the fraction of generated tokens sharing an identical replayable topology. Require >=95% within a selected steady-state lane before implementing retained replay.
2. Build the minimum public-API baseline first: HIP graph/stream capture or existing llama.cpp graph execution with all graph-visible arenas pre-grown. Copy NInfer/Gufo's resource-preparation rule: warm a shape eagerly, reserve stable KV/scratch resources, then capture; never allocate/grow a graph-visible arena during replay.
3. Measure public HIP graph against normal scheduler using the exact QFP13 profile-v2 lane. If launch gaps remain large and Hipfire's lower-level transport is still plausibly relevant, prototype retained packet/PM4 submission behind an experimental build flag. Keep it isolated from normal backend code and fail closed when topology/pointers change.
4. Replay validity key should be compact and generation-based, not a giant tensor scan. Conceptual shape:

```cpp
struct bc_decode_replay_key {
    uint64_t topology_hash;
    uint64_t pointer_generation;
    uint32_t n_tokens;
    uint32_t kv_bucket;
    uint16_t n_seq;
    uint8_t  mtp_depth;
    uint8_t  flags;
};
```

A mismatch falls back to the ordinary scheduler and may recapture after a bounded warm count. No replay path may silently patch an address whose allocation generation changed.
5. Keep synchronization minimization separate and evidence-based. A wait/fence can only be removed when producer/consumer buffer ranges prove independence. Hipfire's +7.5% number is a combined retained-replay + one-wait result; BigCherry must report transport-only and fence-only deltas separately.
6. Add NInfer/Gufo graph-resource preparation as a dependency of the replay experiment, not a new cache policy. QFP06 remains superseded: fixed caps below the live graph working set caused recapture churn and are not reconsidered here.
7. Route Splash lifetime planning to the existing memory/workspace owner if one exists. Before creating any allocator patch, inspect ggml scheduler allocation/liveness: only add an item for proven simultaneously allocated scratch buffers whose lifetimes do not overlap and whose alignment/storage class permit aliasing.
8. Route DwarfStar expert persistence/prefetch and llama.cpp #29887 host-expert LRU to MET01/MET*. Use their policy patterns, not independent stores. On this no-P2P discrete system, prioritize stable whole-expert/static-hot placement plus a bounded host-upload tail; asynchronous prefetch is useful only if route lead time hides PCIe/host latency.
9. Route llamAmpere multi-column reuse to the current MTP/MMVQ owner. Benchmark verification widths 2/3/4/5/6/7/8 and capture weight-unpack/dequant work per accepted token. If the same quant block is decoded once and reused across columns without extra VGPR spills, fold the specialization into the existing kernel; do not add a parallel verify kernel family unless resource evidence requires it.
10. Add a **Strata ceiling gate** before kernel promotion for host/SSD-streamed MoE paths: measure wall-time share of H2D/staging, host expert work, synchronization and target GPU compute. Estimated maximum end-to-end gain from optimizing a kernel with fraction `f` is bounded by `1/(1-f)` even with infinite kernel speedup. If that ceiling is below the project acceptance threshold, close/deprioritize the kernel experiment and optimize residency/transport instead.
11. If "rata" referred to a different canonical project than Strata, add it only after its exact public repo/source is identified. Mechanism promotion rule remains: exact source seam + independent benchmark + existing-owner check before a new plan may be created.

## Detailed Solution & Technical Design

### A. Retained replay owner

This item owns only submission/replay transport. QFP13 remains the launch census/ranking/acceptance umbrella; PRBE37/38/39/40 and Q8_1 producer/cache items continue to own kernel/node elimination; QFP11/RNX11 continue to own split/collective boundaries. Retained replay composes after those optimizations and must not duplicate them.

Capture flow:

```cpp
if (!replay.matches(key)) {
    execute_normal();
    warm_resource_arenas();
    replay.observe(key, launch_trace);
    if (replay.stable_enough(key)) replay.capture(key);
} else {
    replay.submit(key);
}
```

The first implementation should contain no model-specific kernel logic. It records/submits an already-qualified sequence. This keeps the code smaller than creating fused mega-kernels for every residual launch and lets QFP13 continue reducing the tape independently.

### B. Capture-safe resource lifetime

NInfer's Qwen graph preparation reserves DFlash capture rows/KV address-space handles before capture. Gufo explicitly runs each batch shape eagerly before capture so its GEMM arena can grow first. BigCherry should adopt the same invariant: all replay-visible device addresses are stable for a replay generation. Arena growth increments `pointer_generation` and invalidates retained executions.

### C. Multi-column verification reuse

llamAmpere's useful mechanism is not Ampere-specific arithmetic; it is avoiding repeated decode/unpack work across speculative verification columns. For MMVQ-style verification, structure the inner work so a quant block's metadata/scales/dequantized lane fragment is loaded once, then accumulated into several output columns. Gate on compiler-resource evidence: reuse that raises VGPR pressure enough to reduce occupancy or spill is rejected. This should be evaluated first on gfx1201 R9700 where MTP runs on a separate card, then gfx1100.

### D. Residency, memory planning, and bottleneck ceilings

DwarfStar, Splash, llama.cpp #29887 and Strata all reinforce policy/lifetime separation. Expert residency policy should output placement/cache/prefetch decisions consumed by existing loaders/transports. Workspace liveness should produce aliases consumed by the existing allocator. Neither should introduce a second allocator or second expert store.

Before porting a faster compute path, calculate a measured ceiling:

```text
f_compute = target_kernel_ms / end_to_end_step_ms
max_speedup_if_kernel_free = 1 / (1 - f_compute)
```

This is deliberately simple. It prevents a recurring anti-pattern where a strong isolated GEMM/MMQ benchmark is promoted into a streaming-MoE configuration whose wall clock is dominated by host/PCIe staging. Use profiler totals, not theoretical bandwidth, for `f_compute`.

### E. Current llama.cpp digest routing

The 2026-10-04 upstream scan is consolidated rather than duplicated:
- #29910 Q2_K MMQ spill elimination remains owned by PEF06/existing Q2_K MMQ line; replace local overlapping Q2_K code if upstream wins.
- #29887 host-resident expert LRU is owned by MET01 and must be compared at equal VRAM against static/hybrid placement.
- #29924 n-gram truncation candidate-state fix is owned by PRBE52/spec validation because mixed `ngram-mod,draft-mtp` performance can otherwise be misdiagnosed.
- merged #29825 QSA score-memory compaction is the upstream baseline in QFP04 before further sparse-attention workspace work.

## Code Samples & Guidance

Retained execution metadata must stay out of kernel hot paths. Store immutable launch descriptors plus a small patch table for legal dynamic scalar fields only. Pointer patching is allowed only for allocations with the same generation/size/alignment contract; otherwise invalidate.

Do not copy Hipfire private HSA/PM4 machinery until the public HIP-graph baseline is measured. If PM4 is tested, isolate it in an experimental provider/source file so upstream rebases can drop it cleanly.

For verify-width reuse, prefer a compile-time small width specialization (`W=2..8`) selected outside the dot-product loop. Do not branch per column inside the quant inner loop.

For streaming MoE, attach the measured ceiling fields to evidence rows: `{step_ms, staging_ms, host_expert_ms, gpu_moe_ms, sync_ms, theoretical_kernel_free_speedup}`. A candidate below threshold should be retired before source work begins.

## Files

New retained-replay experiment/provider files only after Stage 0 proves stability; existing graph/scheduler/backend sources at the current llama.cpp pin; existing MTP/MMVQ owner files for verify-width work; existing MET placement/prefetch tooling; existing allocator/workspace owner for Splash-style liveness experiments. No Strata-specific runtime file is imported: only evidence/ceiling methodology is reused.

## Validation

Primary hardware: R9700 gfx1201 and one RX 7900 XTX gfx1100; then 2x XTX tensor split. Production model lanes: Flash-Next and dense Qwen3.8-27B controls. Measure normal scheduler vs HIP graph vs retained replay. Record launches/token, host submit CPU time, same-stream gap ms/token, GPU busy share, ms/token, effective TG, replay hit/miss/invalidation counts, and bit-exact greedy output.

Retained replay acceptance: >=3% end-to-end TG improvement or >=0.15 ms/token serial submit/gap reduction with no correctness regression and no >1% regression on topology-miss fallback. PM4/private-runtime code is not accepted into the default path unless it materially beats the public graph baseline and is isolated behind a build/runtime gate.

Verification-reuse acceptance: lower dequant/unpack instruction work at width >=2, no new scratch spill, no >1% single-token decode regression, and improved effective MTP throughput/accepted-token latency on at least one production width.

Strata ceiling gate: every proposed compute-kernel optimization on a host/SSD-streamed expert lane records current wall-time fraction and maximum possible end-to-end gain. If even an impossible zero-time kernel cannot meet the normal acceptance threshold, route the work to MET residency/transport instead.

## Effort & Risk

High for retained PM4; medium for public graph resource preparation and verify-width reuse. Main risks are pointer lifetime, capture invalidation, ROCm/HSA version coupling, hidden serialization, and optimizing a non-dominant compute component. This is why PM4 is a second-stage experiment and why the ceiling gate precedes streaming-MoE kernel work.

## Standards

Reuse-before-fork; one canonical owner per source seam; fail-closed replay validity; production-LOC gate; measured bottleneck/ceiling before kernel work; existing deterministic correctness/evidence contracts.

## Acceptance Criteria

- Stage 0 proves a stable replayable decode topology or closes the retained-submission path with evidence.
- Public HIP graph baseline uses pre-grown/stable resources and is measured before private PM4 work.
- Any retained replay reports transport-only performance separately from synchronization removal.
- DwarfStar/Splash/NInfer/Gufo/llamAmpere/Strata mechanisms are assigned to existing owners where applicable; no duplicate allocator, expert cache, graph cap, MTP kernel family or attention selector is introduced.
- Streaming-MoE kernel experiments pass the measured ceiling gate before implementation.
- #29910/#29887/#29924/#29825 are routed to PEF06/MET01/PRBE52/QFP04 respectively rather than spawning parallel plan lines.
- Production code grows only for a measured mechanism that cannot be expressed by existing owner infrastructure; superseded/duplicate code is deleted when an external mechanism replaces it.

## Notes

Verified references (2026-10-04):
- Hipfire retained dispatch/R9700 evidence: https://github.com/warpfront/hipfire/blob/a89ed0a8e9d8dc7a22d4e6dcbacd57bf2abfad74/crates/redline-dispatch/HIPFIRE-GRAFT.md
- Hipfire architecture/retained replay: https://github.com/warpfront/hipfire/blob/master/docs/ARCHITECTURE.md
- NInfer Qwen graph preparation: https://github.com/Neroued/ninfer/blob/d44ab58408aa389728cd8b1ee50179527e1f3e0d/src/models/qwen3_5/program/graphs.cpp
- Gufo HIP graph warm/capture executor: https://github.com/gufo-org/gufo/blob/8bdde807e57fadfe57f4a1005707559ae6afc82f/src/models/qwen38_flash_next/kernels/rocm/executor.cpp
- Splash runtime/memory planning: https://github.com/incoai/splash/blob/604c70f14c6dca0cd792b01d780757b1c8c5bab8/README.md
- Splash staged state lifetime: https://github.com/incoai/splash/blob/604c70f14c6dca0cd792b01d780757b1c8c5bab8/runtime/model/QwenState.hpp
- DwarfStar expert cache/prefetch source: https://github.com/antirez/ds4/blob/0aaea5a238fb41a35106a551e73c8409dfb751ac/ds4_gpu.h
- llamAmpere speculative width reuse: https://github.com/JakeATX/llamAmpere/blob/3430447a5fddded9c6d0d8cdc714fdb51c04fab5/docs/bonsai2.md
- llamAmpere Qwen/verify attention specialization: https://github.com/JakeATX/llamAmpere/blob/3430447a5fddded9c6d0d8cdc714fdb51c04fab5/README.md
- Strata C++/CUDA MoE engine and bottleneck/placement evidence: https://github.com/ro99/strata
- llama.cpp Q2_K spill PR #29910: https://github.com/ggml-org/llama.cpp/pull/29910
- llama.cpp host-expert GPU LRU PR #29887: https://github.com/ggml-org/llama.cpp/pull/29887
- llama.cpp speculative truncation fix PR #29924: https://github.com/ggml-org/llama.cpp/pull/29924
- llama.cpp merged QSA memory PR #29825: https://github.com/ggml-org/llama.cpp/pull/29825

Current BigCherry owners checked before creation: QFP13 launch-ranking umbrella; QFP06 superseded graph-cap evidence; QFP11/RNX11 collective/split boundary; MET expert tiering; existing MTP/MMVQ/FA and memory owners. PEF07 exists because retained submission transport does not have an existing canonical owner.

## Change Log

- 2026-10-04T00:30:00+00:00 (agent): Created after cross-engine source scan; promoted Hipfire-style retained submission as a new owner and mapped NInfer/Gufo/Splash/DwarfStar/llamAmpere mechanisms to existing owners.
- 2026-10-04 (agent): Resolved requested "rata" as likely Strata, added measured bottleneck-ceiling gating for streamed MoE, and routed the current llama.cpp #29910/#29887/#29924/#29825 digest to canonical owners.

## Ledger-events

