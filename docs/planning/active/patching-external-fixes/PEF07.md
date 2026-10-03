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
- **Splash** combines specialized kernels with automatic memory planning; its Qwen state staging makes an async/disk writer consume a staging allocation so the source state's buffers return to the pool immediately. Import lifetime-based workspace reuse, not Metal code.
- **DwarfStar/ds4** keeps its resident SSD expert cache warm across sessions and has optional expert look-ahead/prefetch machinery. Residency/prefetch policy belongs to MET, not a second expert-cache subsystem.
- **llamAmpere** specializes speculative verification widths: its ternary verify path unpacks a weight block once per lane and dots multiple verification columns, instead of re-unpacking per column; its attention path separately specializes verify widths 6-8. Import multi-column reuse/verify-width specialization into the existing MTP/MMVQ/FA owners, not SM86 CUDA kernels.
- **Rata** could not be resolved to a unique public inference-engine repository in this scan. Do not create a plan from an ambiguous name; add it only when its canonical repo/source is identified.

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
8. Route DwarfStar expert persistence/prefetch to MET01/MET*; use its policy pattern, not SSD transport. On this no-P2P discrete system, prioritize stable whole-expert GPU placement and host-pinned fallback; asynchronous prefetch is only useful if routing look-ahead is early enough to hide PCIe/host latency.
9. Route llamAmpere multi-column reuse to the current MTP/MMVQ owner. Benchmark verification widths 2/3/4/5/6/7/8 and capture weight-unpack/dequant work per accepted token. If the same quant block is decoded once and reused across columns without extra VGPR spills, fold the specialization into the existing kernel; do not add a parallel verify kernel family unless resource evidence requires it.
10. Re-run the external scan when canonical Rata/other engine repositories are identified. Mechanism promotion rule: exact source seam + independent benchmark + existing-owner check before a new plan may be created.

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

### D. Residency and memory planning

DwarfStar and Splash both separate policy/lifetime from compute. Preserve that separation. Expert residency policy should output placement/prefetch decisions consumed by existing loaders/transports. Workspace liveness should produce aliases consumed by the existing allocator. Neither should introduce a second allocator or second expert store.

## Code Samples & Guidance

Retained execution metadata must stay out of kernel hot paths. Store immutable launch descriptors plus a small patch table for legal dynamic scalar fields only. Pointer patching is allowed only for allocations with the same generation/size/alignment contract; otherwise invalidate.

Do not copy Hipfire private HSA/PM4 machinery until the public HIP-graph baseline is measured. If PM4 is tested, isolate it in an experimental provider/source file so upstream rebases can drop it cleanly.

For verify-width reuse, prefer a compile-time small width specialization (`W=2..8`) selected outside the dot-product loop. Do not branch per column inside the quant inner loop.

## Files

New retained-replay experiment/provider files only after Stage 0 proves stability; existing graph/scheduler/backend sources at the current llama.cpp pin; existing MTP/MMVQ owner files for verify-width work; existing MET placement/prefetch tooling; existing allocator/workspace owner for Splash-style liveness experiments.

## Validation

Primary hardware: R9700 gfx1201 and one RX 7900 XTX gfx1100; then 2x XTX tensor split. Production model lanes: Flash-Next and dense Qwen3.8-27B controls. Measure normal scheduler vs HIP graph vs retained replay. Record launches/token, host submit CPU time, same-stream gap ms/token, GPU busy share, ms/token, effective TG, replay hit/miss/invalidation counts, and bit-exact greedy output.

Retained replay acceptance: >=3% end-to-end TG improvement or >=0.15 ms/token serial submit/gap reduction with no correctness regression and no >1% regression on topology-miss fallback. PM4/private-runtime code is not accepted into the default path unless it materially beats the public graph baseline and is isolated behind a build/runtime gate.

Verification-reuse acceptance: lower dequant/unpack instruction work at width >=2, no new scratch spill, no >1% single-token decode regression, and improved effective MTP throughput/accepted-token latency on at least one production width.

## Effort & Risk

High for retained PM4; medium for public graph resource preparation and verify-width reuse. Main risks are pointer lifetime, capture invalidation, ROCm/HSA version coupling, and hidden serialization. This is why PM4 is a second-stage experiment rather than the first implementation.

## Standards

Reuse-before-fork; one canonical owner per source seam; fail-closed replay validity; production-LOC gate; existing deterministic correctness/evidence contracts.

## Acceptance Criteria

- Stage 0 proves a stable replayable decode topology or closes the retained-submission path with evidence.
- Public HIP graph baseline uses pre-grown/stable resources and is measured before private PM4 work.
- Any retained replay reports transport-only performance separately from synchronization removal.
- DwarfStar/Splash/NInfer/Gufo/llamAmpere mechanisms are assigned to existing owners where applicable; no duplicate allocator, expert cache, graph cap, MTP kernel family or attention selector is introduced.
- Production code grows only for a measured mechanism that cannot be expressed by existing owner infrastructure; superseded/duplicate code is deleted when an external mechanism replaces it.

## Notes

Verified references (2026-10-04):
- Hipfire retained dispatch/R9700 evidence: https://github.com/warpfront/hipfire/blob/a89ed0a8e9d8dc7a22d4e6dcbacd57bf2abfad74/crates/redline-dispatch/HIPFIRE-GRAFT.md
- NInfer Qwen graph preparation: https://github.com/Neroued/ninfer/blob/d44ab58408aa389728cd8b1ee50179527e1f3e0d/src/models/qwen3_5/program/graphs.cpp
- Gufo HIP graph warm/capture executor: https://github.com/gufo-org/gufo/blob/8bdde807e57fadfe57f4a1005707559ae6afc82f/src/models/qwen38_flash_next/kernels/rocm/executor.cpp
- Splash runtime/memory planning: https://github.com/incoai/splash/blob/604c70f14c6dca0cd792b01d780757b1c8c5bab8/README.md
- Splash staged state lifetime: https://github.com/incoai/splash/blob/604c70f14c6dca0cd792b01d780757b1c8c5bab8/runtime/model/QwenState.hpp
- DwarfStar expert cache/prefetch source: https://github.com/antirez/ds4/blob/0aaea5a238fb41a35106a551e73c8409dfb751ac/ds4_gpu.h
- llamAmpere speculative width reuse: https://github.com/JakeATX/llamAmpere/blob/3430447a5fddded9c6d0d8cdc714fdb51c04fab5/docs/bonsai2.md
- llamAmpere Qwen/verify attention specialization: https://github.com/JakeATX/llamAmpere/blob/3430447a5fddded9c6d0d8cdc714fdb51c04fab5/README.md

Current BigCherry owners checked before creation: QFP13 launch-ranking umbrella; QFP06 superseded graph-cap evidence; QFP11/RNX11 collective/split boundary; MET expert tiering; existing MTP/MMVQ/FA and memory owners. PEF07 exists because retained submission transport does not have an existing canonical owner.

## Change Log

- 2026-10-04T00:30:00+00:00 (agent): Created after cross-engine source scan; promoted Hipfire-style retained submission as a new owner and mapped NInfer/Gufo/Splash/DwarfStar/llamAmpere mechanisms to existing owners.

## Ledger-events

