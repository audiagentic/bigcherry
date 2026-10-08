---
id: QFP46
order: 46
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-08T22:20:00+11:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: M
---

# Prefill concurrency contracts: critical-path DAG, stream/event ownership, and deterministic model tests

## Description

Audit of Strata-inspired portability at b11474 shows no justification for a second inference/prefill engine yet. BigCherry already owns the major mechanisms: QFP17 (prefill attribution, ubatch/QSA), QFP24/1326 (large pinned host input ring), QFP37/QFP44 (MMQ occupancy), QFP41 (persistent per-device host dispatch), QFP42/1348 (asynchronous NextN and deferred MTP), PGC15 (token-tiled compute/RCCL AllReduce), PGC16 (only provably foldable reductions), QFP45 (large-batch and reference-stack comparison), MET05/10 (auxiliary backend/nonresident expert transfer), and RPL01 (read-only topology/placement model).

This item owns ONLY the common **dependency/critical-path contract**, composability tests, and evidence format. It does not own a new scheduler, a new collective API, a new expert-cache implementation, or an MMQ rewrite. Avoid claiming parallelism merely because independent streams or threads exist; prove overlapping device intervals and a shorter exposed critical path.

## Steps

1. Use existing 1319/1320/1325/1346 diagnostics and rocprofv3 HIP API/kernel/memcpy activity. On pin b11474 with production 2x gfx1100 + gfx1201 target and gfx1030 draft, profile 8K/24K/98K uncached Flash-Next IQ4_XS at ub512; compare ub1024 where it fits with QFP17's existing QSA chunk. Include a no-draft control and Qwen3.8-27B Q8_0. Attribute blocked host time separately from GPU device duration; align rank arrival times rather than summing kernel times across ranks.
2. Build a read-only DAG per prompt chunk and per device: host graph build, input set, Meta split submission, target layer compute, attention, routed MoE, AllReduce producer/rank arrival/completion, NextN output ownership, staging D2H event, MTP draft catch-up, and final flush. Identify a *real* dependency for every proposed synchronization removal.
3. For each candidate, state what can run independently and what cannot. Record measured exposed wall-time upper bound, synchronization/copy volume, allocator pressure and graph-capture constraints. Rank by removable critical-path time, not kernel utilization alone.
4. Maintain a narrow shared trace join key: request id + target/draft context + prompt chunk + Meta graph UID + rank/device + producer/tile ordinal + event generation. Implement extensions INSIDE owning diagnostics patches, not a second tracing stack. Clock domains must be correlated before claiming cross-device overlap.
5. Run the deterministic mock contract suite at tools/tests/prefill/test_prefill_contract_model.py. It models event lifetime and ordering, not GPU performance. Turn every real failure/race from QFP41/QFP42/PGC15/QFP24/MET12/QFP47 into a case in each owner's production-focused test suite.
6. Run hardware A/B arms independently, then the profitable composition with allocation peak, target-authoritative greedy identity, near-tie reference probes, decode control, prompt-cache reuse, cancel/context shift, and 2 concurrent server slots.

## Detailed Solution & Technical Design

### Allowed overlap and true barriers

| Lane | Permitted concurrency | Dependency that must remain | Implementation owner |
|---|---|---|---|
| Target/host | Host prepares chunk k+1 while GPU computes k when input and graph state lifetimes permit | CPU cannot overwrite graph/input storage used by k | QFP42 with 1326/QFP24 |
| Draft/target | Draft catch-up k while target k+1 executes after NextN k has been staged | Draft k consumes exact completed NextN k; final flush precedes sampling | QFP42 and 1348 |
| Per-device host dispatch | One persistent worker per rank submits independent device work | All ranks retain the same collective ordinal and graph UID, with joins | QFP41 |
| Tiled target/collective | Produce tile i+1 while reducing complete tile i | Every rank's exact tile must be ready; consumers wait for reduced data | PGC15 |
| Host expert weights/compute | Prefetch predicted hot weights while unrelated earlier work executes | Exact routed misses must arrive before their MoE compute; within-layer route depends on preceding attention | MET12 (conditional on over-VRAM) |
| Prefill/decode scratch | Borrow eligible evictable cache space during prefill | Restore all borrowed state and retire in-flight events before decode | QFP47 |
| MoE GEMM occupancy | Wider eligible tile/Stream-K distribution | Respect per-rank expert range, reduction order and fallbacks | QFP37/QFP44 |

A GPU can be busy but stalled on memory bandwidth or collectives; no amount of extra host threads fixes a fully occupied device. Similarly, a host blocked in a true NextN producer dependence cannot be made asynchronous by reordering a wait without owning output lifetime.

### Minimal trace contract

Emit an opt-in trace whose fields identify: request, phase, context, graph_uid, chunk, rank, operation, dependency/event_generation, enqueue_ns, device_start_ns, device_end_ns, host_wait_ns, bytes, buffer_owner and fallback_reason. Time bases are identified explicitly; never directly subtract unrelated HIP and host clocks. Emit bounded aggregated per-chunk counters when full tracing would perturb timing.

### Go/no-go

- **No generic prefill rewrite** until at least two independent targeted improvements have been qualified and residual GPU idle/submission overhead is still exposed and attributable.
- Reject a candidate if the timeline shows no overlap or gains are entirely due to changed quantization, cache state, graph shape, workload or MTP acceptance.
- Preserve a same-binary default-off arm and fail closed on absent backend events, unsupported Meta ownership, graph capture, tensor strides, or ambiguous rank ordering.
- Report measured upper bounds before estimated E2E gains. No performance claim is made by the mock tests.

## Parallel execution map (one slice/branch/PR per production mechanism)

- **Lane A, now:** QFP46 instrumentation contract + pure-Python oracle. Runs without Brutus; does not gate design work in the other lanes.
- **Lane B, immediate:** QFP42 event-owned NextN staging, after PR #8 fixes the deferred-path 1346 counters. Independent of C/D/E source changes but avoid editing the same 1326/Meta lines concurrently.
- **Lane C, immediate:** PGC15 range-collective and real producer tiling, after its own RCCL/provider screen. No dependency on QFP42 or QFP41.
- **Lane D, immediate:** QFP41 per-device dispatch thread/ownership design and single-rank event smoke; integrate only after proof that collective order remains stable.
- **Lane E, immediate:** QFP17 QSA-bounded ub1024/2048 fit, QFP37/QFP44 MMQ shape sweeps. These are independent of MTP/AR; physical Brutus validation is queue-serialized.
- **Lane F, conditional:** MET12 over-VRAM expert staging, only if actual H2D critical-path cost is material. Does not alter 1281/1283 expert-range semantics.
- **Lane G, conditional:** QFP47 prefill-only reclaimable scratch lease, only when the QFP17 allocator census proves reclaimable temporary bytes and safe phase boundaries.
- **Integration after lanes:** compare individual default-off changes, then pairs and composed stack on the same build, with repeatability, low/high prompt depth and multi-slot stress. Do not merge code changes into this planning PR.

## Files

- tools/tests/prefill/test_prefill_contract_model.py: standard-library-only synthetic ordering oracle.
- Existing telemetry and hardware tooling stay owned by QFP17, QFP42, QFP41, PGC15 and tools/lab/flash-next.
- Existing branch, release, and adoption rules: docs/reference/build/BRANCHING.md and docs/reference/testing/PATCH_VALIDATION.md.

## Validation

Offline: python -m unittest discover -s tools/tests/prefill -p 'test_*.py' -v. Tests cover a model of host/target/draft overlap, multi-rank event generation reuse, rank/shape/tile consensus, expert miss fallback, and phase scratch restore. These DO NOT exercise real HIP streams, ggml graphs, RCCL, model output or acceleration; those remain owning-slice gates.

Hardware: A/B at 8K/24K/98K + deep long-context, per-rank device timeline, explicit overlapping GPU intervals and exposed wall reduction; output equivalence and lifecycle stress. Record cases where eligible parallelism is zero.

## Acceptance Criteria

1. Every suggested concurrent edge has an owner, producer/consumer lifetime, event/query/fallback and a negative eligibility test.
2. Trace separates host blockage, device busy, rank arrival skew, actual transfer, and hidden asynchronous work.
3. Independent owner branches can proceed in parallel; only real source-level conflicts and shared Brutus queue impose serialization.
4. A real implementation cannot claim success from model-only tests or profiler-summed percentages.

## Change Log

- 2026-10-08: Created after plan deduplication for Strata-style prefill concurrency and mock-contract coverage.
