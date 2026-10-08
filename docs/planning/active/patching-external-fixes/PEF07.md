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

1. Stage 0a (mandatory before topology hashing): distinguish GPU kernel execution from CPU submissions. QFP13's ~1,000-1,300 device kernels/token do NOT imply that many HIP host launches: b11474 already has HIP graph capture/replay. Use `tools/lab/flash-next/graphs-ab.sh` only as a historical multi-GPU graph-on/off control: its `long-ctx-profile.sh` driver hardcodes `HIP_VISIBLE_DEVICES=0,1,2,3`, defaults to tensor split and MTP, and even `NO_MTP=1` retains `-sm tensor`. For the initial single-GPU lane, use a dense Qwen3.8-27B Q4_K_M model that fits a single XTX/R9700 and an isolated ordinary-decode launcher with explicit one-device settings, and trace successful `hipGraphLaunch`, eager launch, capture, instantiate, update and graph-compatibility/warmup-reset events per token/graph key. The existing 1231 HI14 once-per-process markers prove stages occurred but CANNOT provide hit/miss counts; extend that existing evidence path with bounded per-key counters only if required. First qualify single-GPU ordinary decode without MTP/Meta/collectives, then expand only after separate owners clear their lanes.
2. Stage 0b: on the same measured lane, hash ordered GPU launches, kernel identity, grid/block/shared memory, dependency/fence points and pointer/allocation generations; measure >=95% topology stability AND >=95% successful public-HIP-graph replay eligibility. Record both ratios independently, not as interchangeable evidence. If the graph is stable but warmup repeatedly resets, first test upstream #29768 as a guarded public-graph control rather than implementing retained PM4.
3. Reuse the existing public HIP graph baseline (`ggml_cuda_graph_check_compability`, `ggml_cuda_graph_update_required`, `ggml_backend_cuda_graph_compute`, `ggml_cuda_graph_evaluate_and_capture`); do not build a second public graph executor. Pre-grow stable KV/scratch resources before capture. Distinguish graph-disabled eager, stock public graph, and optionally #29768 after pin-compatible mechanics/correctness checks. Leave protected 1340 Meta-arena and MTP work untouched.
4. Measure public HIP graph against graph-disabled eager on QFP13's exact unprofiled request corpus. Account for CPU HIP API time and same-stream GPU gaps separately: a device-kernel gap may remain even when only one `hipGraphLaunch` was submitted. PM4 is ineligible if graph replay is rare (repair eligibility first), if host submission accounts for <5% of end-to-end decode wall time, or if graph-disabled/graph-enabled controls do not isolate the claimed bottleneck. Only then consider a disposable, default-off PM4 experiment, without changing normal backend/Meta/MTP paths.
5. Replay validity key should be compact and generation-based, not a giant tensor scan. Conceptual shape:

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
6. Keep synchronization minimization separate and evidence-based. A wait/fence can only be removed when producer/consumer buffer ranges prove independence. Hipfire's +7.5% number is a combined retained-replay + one-wait result; BigCherry must report transport-only and fence-only deltas separately.
7. Add NInfer/Gufo graph-resource preparation as a dependency of the replay experiment, not a new cache policy. QFP06 remains superseded: fixed caps below the live graph working set caused recapture churn and are not reconsidered here.
8. Route Splash lifetime planning to the existing memory/workspace owner if one exists. Before creating any allocator patch, inspect ggml scheduler allocation/liveness: only add an item for proven simultaneously allocated scratch buffers whose lifetimes do not overlap and whose alignment/storage class permit aliasing.
9. Route DwarfStar expert persistence/prefetch and llama.cpp #29887 host-expert LRU to MET01/MET*. Use their policy patterns, not independent stores. On this no-P2P discrete system, prioritize stable whole-expert/static-hot placement plus a bounded host-upload tail; asynchronous prefetch is useful only if route lead time hides PCIe/host latency.
10. Route llamAmpere multi-column reuse to the current MTP/MMVQ owner. Benchmark verification widths 2/3/4/5/6/7/8 and capture weight-unpack/dequant work per accepted token. If the same quant block is decoded once and reused across columns without extra VGPR spills, fold the specialization into the existing kernel; do not add a parallel verify kernel family unless resource evidence requires it.
11. Add a **Strata ceiling gate** before kernel promotion for host/SSD-streamed MoE paths: measure wall-time share of H2D/staging, host expert work, synchronization and target GPU compute. Estimated maximum end-to-end gain from optimizing a kernel with fraction `f` is bounded by `1/(1-f)` even with infinite kernel speedup. If that ceiling is below the project acceptance threshold, close/deprioritize the kernel experiment and optimize residency/transport instead.
12. If "rata" referred to a different canonical project than Strata, add it only after its exact public repo/source is identified. Mechanism promotion rule remains: exact source seam + independent benchmark + existing-owner check before a new plan may be created.

## Detailed Solution & Technical Design

### A. Retained replay owner

This item owns only submission/replay transport. QFP13 remains the launch census/ranking/acceptance umbrella; PRBE37/38/39/40 and Q8_1 producer/cache items continue to own kernel/node elimination; QFP11/RNX11 continue to own split/collective boundaries. Retained replay composes after those optimizations and must not duplicate them.

Capture/admission flow (conceptual; reuse upstream's graph executor rather than adding this as another cache):

```cpp
prepare_stable_resources_before_capture();
const auto actual = observe_existing_hip_graph_path(); // eager / capture / replay / incompatible
if (!actual.replay_eligible || actual.pointer_generation_changed || actual.topology_changed) {
    use_existing_scheduler(); // preserve the existing graph warmup and fail-closed fallback
} else if (public_graph_is_qualified && measured_host_submit_share >= 0.05) {
    qualify_optional_retained_transport_against_public_graph(); // no production PM4 yet
}
```

A `ggml_cgraph::uid` or first-node pointer is not a portable lifetime generation. Preserve the upstream per-node property comparison and backend-context ownership; do not resurrect rejected 1233 shape-key hashing or 1304 graph-cache caps.

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

### F. b11474 HIP graph admission correction (2026-10-08)

**Pinned source, not a hardware measurement.** At llama.cpp b11474 (b9acf138), `ggml/src/ggml-cuda/ggml-cuda.cu` already implements the public HIP graph path: `ggml_cuda_graph_check_compability` (2644-2675), `ggml_cuda_graph_get_key` (2677-2679), `ggml_cuda_graph_update_required` (2681-2721), `ggml_cuda_graph_evaluate_and_capture` (4312+, `cudaGraphLaunch` at 4522), and `ggml_backend_cuda_graph_compute` (4547-4603). `common.cuh` owns the backend-context graph map (1458-1485), ten-second idle eviction, graph properties and `GGML_CUDA_DISABLE_GRAPHS` (1288-1291). A first-node-pointer cache key is only a bucket; node count, op parameters, src data pointers, shapes and strides are compared before reuse. A stable GPU kernel tape is NOT proof of repeated host submissions or graph-cache hits.

**Observed control flow in source:** a changed node/property set runs eagerly until a subsequent stable call captures; after successful graph replay, another property change resets warmup and again executes eagerly. This can cost two host-dispatched calls per shape transition even when the GPU kernel topology remains compatible. No claim is made that this occurs frequently in BigCherry's current production lane. Upstream unmerged [#29768](https://github.com/ggml-org/llama.cpp/pull/29768) changes this policy: after a successful replay and unchanged kernel/non-kernel node topology, recapture on the next property change rather than restarting warmup. Its +1.14% CI95 [+0.44%, +1.84%] result is **RTX 5090**, not RDNA; test only as a disposable public-graph control after real warmup-reset attribution. Unmerged [#29796](https://github.com/ggml-org/llama.cpp/pull/29796) skips redundant sync on **single-device native CUDA only** and explicitly excludes HIP and multi-device: do not port its 52.55% Windows Blackwell claim to this topology.

**Existing BigCherry primitives / negative evidence:** patch `1231_hi14_graph_capture_lifecycle_evidence` has once-per-process begin/end/instantiate/replay markers (not frequency counters); `1302_cuda_graph_oom_evict` is validated HIP-only OOM recovery; `1304_cuda_graph_lru_cap` was rejected after ~195 live split graphs made small caps recapture repeatedly; `1233_rd73_stable_graph_cache_key` was demoted/rejected after balanced gfx1100 tests (tg128 -1.96%, tg512 -1.24%, pp1024 -0.69%). Reuse these ownership/evidence paths. Never reintroduce the shape-hash key, small cache cap, or a second graph allocator.

**Cheapest discriminator:** use an isolated one-device launcher (do not run `graphs-ab.sh` unchanged: its underlying `long-ctx-profile.sh` hardcodes a four-device visibility list and tensor-split options) on a clean, single-GPU dense-27B plain-decode lane (8K and 80K context; no MTP, Meta or cross-device collectives); separately preserve Flash-Next production as a later noninterfering control. Record, per generated token and graph key: eager dispatches, `hipGraphLaunch`, capture begin/end, instantiate/update, property-change reset, compatibility rejection, graph eviction/OOM recovery, and CPU HIP API submit/sync time. Compare GPU kernel count and same-stream gaps separately; do not infer CPU launch count from rocprof kernel rows. If >95% of steps are already graph replays and host submit/sync <5% of wall time, **close PM4 for that lane**. If resets dominate, test #29768 as an isolated, default-off source diff before PM4. If graphs are incompatible, investigate the exact unsupported op/stream transition with its existing owner rather than bypass capture safety. For all arms verify equal work/token, stable allocation generation, repeated requests and bit-exact output; a faster arm with missing work fails.

**Fork comparison:** Hipfire's [graft report](https://github.com/warpfront/hipfire/blob/a89ed0a8e9d8dc7a22d4e6dcbacd57bf2abfad74/crates/redline-dispatch/HIPFIRE-GRAFT.md) retains an 833-launch / 26-unique-kernel Qwen3.6 A3B tape and reports 164.220→174.087 tok/s against its own HipGraph, with a separate wait-elision combination at 165.839→178.320 tok/s. This is external gfx1201 engine evidence, not BigCherry performance. Gufo's [HIP graph executor](https://github.com/gufo-org/gufo/blob/8bdde807e57fadfe57f4a1005707559ae6afc82f/src/models/qwen38_flash_next/kernels/rocm/executor.cpp) at `Executor::Run` eagerly warms the shape before capture and invalidates graph executables on state/rope changes; reuse its lifetime principle, not its runtime.

**Ownership:** PEF07 only owns any proven retained *submission transport* gap. QFP13 owns GPU kernel/launch ranking and measured end-to-end gate; HI14/1231 owns lifecycle observation; upstream HIP graph owns capture/replay; 1302 owns OOM recovery; QFP06/1304 owns rejected cap evidence; RD73/1233 owns rejected shape-key evidence. PA47/1340, QFP37 and QFP43/MTP work updated within 12 hours is protected: no edits, new experiments, or hardware queue contention here. Follow up with those owners only after their work is terminal.

## Code Samples & Guidance

Retained execution metadata must stay out of kernel hot paths. Store immutable launch descriptors plus a small patch table for legal dynamic scalar fields only. Pointer patching is allowed only for allocations with the same generation/size/alignment contract; otherwise invalidate.

Do not copy Hipfire private HSA/PM4 machinery until the public HIP-graph baseline is measured. If PM4 is tested, isolate it in an experimental provider/source file so upstream rebases can drop it cleanly.

For verify-width reuse, prefer a compile-time small width specialization (`W=2..8`) selected outside the dot-product loop. Do not branch per column inside the quant inner loop.

For streaming MoE, attach the measured ceiling fields to evidence rows: `{step_ms, staging_ms, host_expert_ms, gpu_moe_ms, sync_ms, theoretical_kernel_free_speedup}`. A candidate below threshold should be retired before source work begins.

## Files

New retained-replay experiment/provider files only after Stage 0 proves stability; existing graph/scheduler/backend sources at the current llama.cpp pin; existing MTP/MMVQ owner files for verify-width work; existing MET placement/prefetch tooling; existing allocator/workspace owner for Splash-style liveness experiments. No Strata-specific runtime file is imported: only evidence/ceiling methodology is reused.

## Validation

Primary hardware: R9700 gfx1201 and one RX 7900 XTX gfx1100; then 2x XTX tensor split. Production model lanes: Flash-Next and dense Qwen3.8-27B controls. Measure normal scheduler vs HIP graph vs retained replay. Record launches/token, host submit CPU time, same-stream gap ms/token, GPU busy share, ms/token, effective TG, replay hit/miss/invalidation counts, and bit-exact greedy output.

Retained replay acceptance: >=4 independent sessions, >=10 order-balanced paired rounds/session, bit-identical greedy/logits/recurrent/KV outputs across multi-request same-process and capture invalidations, CI95-low >=3% end-to-end TG improvement versus the *already enabled* public HIP graph, <=1% control/fallback regression, and no memory/resource leak. >=0.15 ms/token submit/gap reduction is diagnostic only, never sufficient alone. Explicitly isolate transport-only from any synchronization removal. PM4/private-runtime code remains default-off and may be rejected solely on maintenance/runtime-version coupling.

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

- 2026-10-08 (BCOP68): Corrected retained-replay admission for b11474's existing HIP graph, pinned warmup/recapture control flow, #29768 comparison, and rejected 1233/1304 negative evidence; no runtime changes or hardware claim.

- 2026-10-04T00:30:00+00:00 (agent): Created after cross-engine source scan; promoted Hipfire-style retained submission as a new owner and mapped NInfer/Gufo/Splash/DwarfStar/llamAmpere mechanisms to existing owners.
- 2026-10-04 (agent): Resolved requested "rata" as likely Strata, added measured bottleneck-ceiling gating for streamed MoE, and routed the current llama.cpp #29910/#29887/#29924/#29825 digest to canonical owners.

## Ledger-events

