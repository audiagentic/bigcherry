---
id: RPL01
order: 1
plan: patching-runtime-placement
state: pending
created-at: '2026-10-06T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Topology-aware runtime placement and execution cost model

## Description

Create one read-only decision layer that combines existing BigCherry topology, profiling, residency and workload evidence before any new runtime scheduler is introduced. BigCherry already has strong local owners (MET01 expert residency, MET05 auxiliary transport, PHA03 topology/RCCL admission, HIP-autotune kernel crossover, llama.cpp split/fit/scheduler machinery), but no authoritative cross-capability model answers the system-level question: for this workload phase and this asymmetric no-P2P topology, where should bytes and work live?

RPL01 owns the cost/evidence model and placement recommendation only. It MUST NOT initially own execution, a second scheduler, a second expert solver, a second collective admission table, or backend-specific transport. Existing owners remain authoritative and consume or constrain RPL01 recommendations.

## Repository evidence and motivation

- MET01 already ranks `(layer,expert,tier)` by avoided critical-path milliseconds per resident byte, but only for routed-expert residency. It explicitly delegates auxiliary transport to MET05 and generic scheduling to upstream llama.cpp.
- PHA03 already owns canonical PCI placement/topology/version evidence and fail-closed RCCL admission. RPL01 must consume this identity rather than rediscover PCIe topology or infer it from HIP ordinals.
- llama.cpp layer split minimizes communication and tensor split performs repeated cross-device reductions; upstream documents tensor split as interconnect-sensitive and gives no AMD performance guarantee. BigCherry therefore cannot treat all visible GPUs as interchangeable workers.
- BigCherry's production target is heterogeneous: gfx1100 XTX devices, gfx1201 R9700, gfx1030 6900 auxiliary device, unequal PCIe links, no normal P2P between primary devices, and workload phases with materially different bottlenecks (prefill, decode, long-context sparse FA, MTP).
- Recent patch 1334 sparse HIP FA evidence materially changes the long-context cost surface: attention cost can fall while placement/transfer/MoE costs remain, so a static placement chosen from older profiles can become wrong even when every local optimisation is individually valid.

## Ownership boundaries

RPL01 consumes, never duplicates:

```text
PHA03/PHA04/PHA05  -> canonical topology, transport capability, RCCL admission
MET01              -> expert residency candidates/budgets and route statistics
MET05/1328         -> 6900 auxiliary execution/staging mechanism
MET02/MET03        -> expert range/sparse execution semantics
HIP-autotune       -> architecture/shape kernel crossover decisions
QFP/FMTP           -> workload-specific FA/MTP measurements
llama.cpp fit/sched-> actual model split, buffer allocation and graph execution
RPL01              -> normalized cost observations + recommendation + counterfactual score
```

No RPL01 code may directly move tensors or launch kernels in Phase A/B. If later evidence justifies runtime actuation, extend the existing llama.cpp/BigCherry owner seam rather than adding an independent scheduler.

## Phase A - canonical observation schema

Add a hardware-free schema/tooling layer under `tools/bigcherry/profiling/` that can ingest existing benchmark/qualification artifacts and emit one normalized observation per phase/device/boundary:

```python
@dataclass(frozen=True)
class PlacementObservation:
    topology_id: str
    workload_id: str
    phase: Literal['prefill', 'decode', 'mtp_verify']
    device: str
    architecture: str
    resident_bytes: int
    peak_vram_bytes: int
    compute_ms: float
    h2d_bytes: int
    d2h_bytes: int
    peer_bytes: int
    transfer_ms: float
    sync_ms: float
    idle_ms: float
    launches: int
    useful_work_units: float
```

Required invariants:

- topology identity comes from the PHA03 evidence schema, not device ordinal;
- measured and estimated fields are distinct and estimates carry provenance;
- byte/time counters are non-negative and physically plausible against measured link bandwidth;
- phase is explicit: prefill evidence cannot silently drive decode placement or vice versa;
- missing transport capability is `unknown` and fail-closed, never assumed available;
- VRAM headroom includes KV/cache/scratch reservations rather than model weights alone.

Phase A does not choose a new split. It proves BigCherry can join existing evidence without inventing another telemetry system.

## Phase B - counterfactual cost model

Implement a deterministic offline scorer over candidate placements already expressible by existing mechanisms. Initial candidates are deliberately bounded:

1. current production layer split;
2. existing explicit tensor/layer split variants already supported by llama.cpp;
3. MET01 expert-residency alternatives;
4. MET05 auxiliary-tier alternative where already qualified;
5. current MTP draft/target placement alternatives already expressible by existing configuration.

For candidate `p`, phase `q`:

```text
T(p,q) = max_device(compute_ms + local_copy_ms)
       + critical_boundary_transfer_ms
       + collective_ms
       + synchronization_ms
       + uncovered_idle_ms
       + capacity_penalty
```

`capacity_penalty = infinity` if model + KV + scratch + cache budgets exceed qualified headroom. Unknown P2P/RCCL capability also makes candidates requiring it inadmissible.

The scorer must report a decomposition, not just a rank, so an agent can see whether a recommendation comes from compute imbalance, transfer, capacity, synchronization or idle time.

Do not fit a sophisticated predictive model initially. Start with measured medians and bounded interpolation. The cheapest discriminator is whether this simple model can rank configurations whose ABBA ordering is already known.

## Phase C - validation against known experiments

Use historical artifacts as hold-out tests. For each available pair/triple:

- hide the measured winner;
- build the cost model from the remaining admissible evidence;
- predict ordering and expected bottleneck;
- compare with actual ABBA/hardware result;
- record ranking accuracy and absolute/relative timing error.

Include at minimum where artifacts exist:

- single vs multi-GPU layer/tensor split;
- dual-XTX vs R9700-related lanes;
- RCCL admitted/rejected topology controls;
- Flash-Next short vs ~100K/~200K prefill including sparse-FA 1334 evidence;
- MTP on/off or draft-placement controls;
- MET expert-residency controls once available.

Promotion to advisory use requires >=80% pairwise winner prediction on qualified hold-outs AND no recommendation that violates a known topology/capacity/correctness gate. If ranking accuracy is below 80%, improve measurement coverage/model decomposition; do not add runtime actuation.

## Phase D - advisory planner only

If Phase C passes, expose an offline command such as:

```text
bigcherry placement recommend --evidence <dir> --workload <profile> --topology <id>
```

Output:

- admissible candidate placements;
- predicted phase cost and decomposition;
- expected peak VRAM/headroom;
- required transport capabilities;
- confidence/provenance;
- exact existing flags/placement JSON needed to reproduce each candidate.

The command must not silently modify runtime configuration. Its first purpose is to choose better ABBA experiments and prevent local optimisation plans from proposing mutually incompatible placements.

## Runtime actuation gate

Do NOT build a dynamic runtime scheduler unless advisory recommendations repeatedly beat the best static production configuration by >=5% end-to-end on at least two materially different workload regimes, with correctness intact, AND the winning recommendation requires a transition that existing configuration cannot express cheaply.

If that gate is reached, the implementation plan must first identify the existing owner seam to extend (llama.cpp fit/scheduler, MET01 placement, or another established owner). A new scheduler is the last resort and requires an explicit duplication review.

## Code-level opportunities this model should expose

The model must make these costs visible rather than hide them in aggregate tok/s:

- redundant H2D/D2H and host-bounce traffic caused by no P2P;
- boundary transfer frequency under layer vs tensor split;
- rank/device arrival skew and uncovered idle;
- KV/scratch/cache VRAM displacement effects;
- persistent-vs-transient expert residency economics;
- whether sparse FA moves the bottleneck from attention to transfer/MoE;
- MTP draft acceptance benefit versus extra target/draft GPU milliseconds;
- auxiliary-device usefulness when its compute gain is smaller than PCIe/PCH staging cost;
- launch/synchronization overhead at small decode batches.

## Validation and correctness

Hardware-free first:

- schema serialization round-trip;
- deterministic candidate ranking;
- missing/unknown topology fails closed;
- impossible VRAM candidates rejected;
- physical bandwidth plausibility checks;
- phase evidence cannot cross-contaminate another phase without explicit fallback/provenance;
- duplicate evidence does not double-count work.

Hardware qualification later uses existing benchmark harnesses; RPL01 does not create another benchmark runner. Require ABBA >=5 repetitions, median/dispersion, multi-request same-process where relevant, long-context lanes, MTP acceptance/effective throughput, and existing logits/greedy correctness contracts.

## Files

Expected initial implementation only:

- `tools/bigcherry/profiling/placement_schema.py` - normalized observation/candidate schema.
- `tools/bigcherry/profiling/placement_cost.py` - deterministic decomposed scorer.
- `tools/tests/profiling/test_placement_schema.py`.
- `tools/tests/profiling/test_placement_cost.py`.
- adapter functions that read existing PHA/MET/QFP evidence; do not fork their schemas.

Do not patch llama.cpp or HIP kernels in Phase A-C.

## External alignment

- llama.cpp's backend scheduler already owns backend assignment, buffer allocation and inter-backend copies; RPL01 should remain an evidence/advisory layer until a measured gap requires extending that owner.
- Current llama.cpp multi-GPU documentation distinguishes low-communication layer split from interconnect-sensitive tensor split and notes RCCL is disabled by default because it was not universally beneficial in testing. This supports a topology-specific admission/cost model rather than a universal multi-GPU policy.
- AMD's upstream tooling work includes per-op profiling/roofline accounting. Reuse/export those measurements where available rather than inventing parallel kernel telemetry.

## Acceptance Criteria

- One normalized placement observation schema consumes existing topology/perf/residency evidence without replacing its authoritative owners.
- Candidate scorer is deterministic, decomposed, capacity-aware and fail-closed for unknown transport capabilities.
- Historical hold-out validation predicts >=80% of qualified pairwise winners before advisory promotion.
- Advisory output maps recommendations to existing flags/placement mechanisms and never mutates runtime state.
- No new scheduler, allocator, transport, expert solver, kernel dispatch table or telemetry collector is introduced in Phase A-C.
- Dynamic actuation remains blocked until >=5% end-to-end benefit is demonstrated in at least two workload regimes and existing configuration cannot express the winning transition.
- Any future runtime implementation extends an established owner seam and passes a fresh duplication/ownership review.

## Notes

This item is intentionally architectural but bounded. The first deliverable is not a scheduler; it is a falsifiable offline model that should make BigCherry's existing optimisations cooperate and choose higher-value experiments. If the simple model cannot predict already-measured winners, close or narrow the effort rather than escalating to a more complex runtime planner.

External references checked 2026-10-06:
- llama.cpp multi-GPU documentation: layer split minimizes communication; tensor split is experimental/interconnect-sensitive; RCCL is not universally beneficial on ROCm.
- llama.cpp `ggml-backend.h`: backend scheduler owns multi-backend allocation/assignment/copies.
- AMD llama.cpp tooling discussion #26333: per-op profiling and roofline accounting intended for HIP/ROCm optimisation.

## Change Log

- 2026-10-06: created after cross-plan audit found no existing system-level placement/cost-model owner; deliberately bounded to observation + offline scoring before any runtime scheduling.


## 2026-10-06 R9700 PCIe-x4 calibration case

Add the current MET01/MET04 measurements as a required Phase-C hold-out. The placement model must predict three qualitative outcomes before advisory promotion: small 2-8 GiB demand caches lose to CPU/no-cache on the ~6.7 GB/s R9700 H2D path; equal-VRAM resident layers beat 8 GiB LRU and dominate prefill near the 12 GiB comparison; and moving expert-bearing layers onto the R9700 as a separate layer-split stage loses materially to production tensor split.

For expert-residency candidates, record measured link bandwidth and miss bytes/token explicitly. Cache hit-rate alone is not a placement cost. The scorer should derive transfer_ms from measured bytes and topology-specific bandwidth when direct timing is unavailable, mark that value estimated, and fail plausibility if implied bandwidth materially exceeds the PHA03/topology measurement.

This does not move cache-policy ownership into RPL01. MET01 still selects expert-residency candidates; RPL01 only checks whether their measured transfer/compute decomposition predicts the observed whole-system ordering.

## 2026-10-10 implementation audit: overlap-safe no-P2P placement scoring (supersedes the Phase-B additive formula)

**Decision:** Do not implement the Phase-B scalar `max_device(compute_ms + local_copy_ms) + critical_boundary_transfer_ms + collective_ms + synchronization_ms + uncovered_idle_ms` as written. It is not an elapsed-time estimator when transfers overlap compute, when a collective includes copy/wait, or when host/device waits already appear in a measured span. An apparently faster placement can be ranked slower. Keep RPL01 **advisory-only** and default to `UNKNOWN` rather than fabricating a winner. This is a cost-model correctness finding, not a measured BigCherry performance regression.

### Pinned implementation and evidence boundaries

- At llama.cpp b11474, `ggml/src/ggml-backend.cpp::ggml_backend_tensor_copy_async` (~651) tries the backend async copy but **synchronizes both source and destination backends before a blocking fallback**. `ggml_backend_sched_compute_splits` (~1853) may wait for the preceding split's event/backend (~1864-1871), then copies inputs (~1874-1885), enqueues split compute (~1887-1889), and records an event (~1926-1929). The wait can already include pending compute/copy; adding `transfer_ms`, `sync_ms` and `idle_ms` as independent serialized terms can count the same elapsed interval multiple times. Upstream master was rechecked 2026-10-10 and retains the same relevant functions; no upstream scheduler change was identified that makes the additive estimator valid.
- `tools/bigcherry/profiling/rocprof.py::parse_kernel_trace` reads Start/End timestamps but aggregates by kernel name into `KernelStat`, discarding the original per-dispatch intervals and request/chunk identity. `tools/bigcherry/profiling/schema.py::KernelStat` stores totals/mean/p95/agent IDs, not dependencies or clock alignment. `tools/lab/flash-next/prefill-kernel-table.py` sums durations per GPU family and divides by a device's min/max kernel span; concurrent kernels may sum to >100% of elapsed busy span. These are useful **hotspot** receipts, not RPL01 critical-path observations. Do not derive per-request wall time by summing them.
- `QFP46` already owns the **shared dependency/critical-path DAG**, trace join identity and event-lifetime contract. Its `tools/tests/prefill/test_prefill_contract_model.py::Timeline/EventRing/TileJoin` are existing hardware-free primitives. RPL01 consumes qualified QFP46/owner-produced event intervals; it must not build a second trace collector, DAG ownership system or GPU event scheduler. QFP41/1356, QFP48/49, PGC15/16 and active Flash-Next experiments remain independent/protected owners.
- No `tools/bigcherry/profiling/placement_schema.py` or `placement_cost.py` exists at the selected branch head. No RPL01 hardware performance or ranking accuracy has been measured. The MET01/MET04 ~6.7 GB/s R9700 H2D observation in this plan is historical calibration evidence, **not** a universal link constant; effective bandwidth varies with transfer size, staging, contention and PCIe route.

### Cheapest discriminator: completed, no hardware

An eight-case disposable Python `unittest` model passed (8/8; no repository modules imported). It proved a concrete ranking inversion:

| Candidate | GPU compute interval (ms) | Copy interval (ms) | Actual makespan | Old additive score |
| --- | --- | --- | ---: | ---: |
| A | [0,8] | [4,10] | **10.0 ms** | 14.0 ms |
| B | [0,10.5] | [10.5,11.5] | 11.5 ms | **11.5 ms** |

The old score picks B even though A completes 1.5 ms sooner. Identical per-kernel totals [8,6] can correspond to 10 ms (overlapped) or 14 ms (serialized); an aggregate-only receipt cannot distinguish them. Additional fixtures covered sync/wait double-counting, mismatched clock domains, unknown/direct-peer rejection, VRAM headroom, and candidate identity. These are synthetic mathematical tests, **not** a compiled GGML test, real trace, benchmark, GPU run or demonstrated production speedup.

### Replacement Phase A/B contract: reuse owner traces, fail closed

1. **Admit a candidate only from existing mechanisms.** Require the same immutable `source_sha/build_plan_id/patch_set/model+GGUF+quant/workload_phase/context/ubatch/request_count/topology_id` and matching backend/ROCm/RCCL provenance for a measured comparison. Device identity is PHA03's canonical PCI/topology identity, never a HIP ordinal alone. Record each observation's owner, raw artifact path/hash, collection mode, units, clock domain, measured-versus-estimated status and confidence. Do not join mismatched request/chunk/graph UID or MTP-acceptance populations.
2. **Treat topology and memory as admission predicates, not additive penalties.** `if unknown_peer_or_rccl_required || direct_peer_without_verified_P2P || peak(weights+KV+scratch+cache+graph+staging)>qualified_budget: INELIGIBLE`. An existing, verified host-staged path is a *different* candidate with its actual measured H2D/D2H/sync cost, not a guessed direct-peer fallback. gfx1030 auxiliary placement must include its chipset route and any CPU staging; dual-XTX/R9700 links and VRAM are not interchangeable.
3. **Use QFP46/owner event receipts when available.** Join only time-aligned per-request, per-phase, per-chunk, per-device `(producer,consumer,event_generation,stream,rank,graph_uid,start,end,bytes)` intervals with known dependencies and lifetime. For each admissible candidate, `elapsed_ms = max(completion_events) - request_phase_start`, or compute the longest path through **the already-owned** QFP46 dependency DAG. Attribute `exposed_compute/exposed_transfer/exposed_collective/exposed_wait` to mutually exclusive wall intervals (union/sweep or causal critical-path ownership); preserve overlapped work as a separate diagnostic, never add it again. Cross-clock alignment uncertainty or absent producer/consumer edges => `UNKNOWN_CRITICAL_PATH`.
4. **Aggregate-only fallback is non-promotable.** Existing `KernelStat` and `prefill-kernel-table.py` may rank hotspots and supply coarse work totals, but cannot establish overlap or score a placement winner. Never impute a zero wait, assume full serialization, or turn an estimated `bytes / measured_bandwidth` into a measured transfer interval. Estimates must be topology/size-qualified, bounded, separately labeled, and may only prioritize a subsequent **existing-owner** measurement.
5. **Counterfactuals must respect changed work.** Before comparing layer/tensor split, expert residency, MTP, sparse FA or auxiliary GPU, require equivalent output, routed expert count, tokens/accepted drafts, KV/graph shape, quant, cache warmup and request concurrency. A different MTP acceptance or skipped reduction is not a placement speedup. Unknown per-rank work/transfer bytes => `UNKNOWN_WORK_EQUIVALENCE`.
6. **No new instrumentation framework.** Initial implementation, if needed, is a small read-only adapter/validator under `tools/bigcherry/profiling/` that consumes existing QFP46 and PHA03/MET evidence. Use `tools/tests/prefill/test_prefill_contract_model.py`'s interval/event semantics as a reference, without editing the active QFP46 path. Do not add a scheduler, cache, solver, allocator, config switch or experiment queue.

### Bounded qualification and terminal decisions

**Gate 0 (no new GPU work):** inventory at least two existing, independently qualified placement A/B pairs with raw event-aligned traces and a measured end-to-end wall clock. If none exist, record `INSUFFICIENT_TIMELINE` and leave the advisory scorer unimplemented; RPL01 can remain a read-only evidence-gap record. A no-P2P/unknown-capability negative control is mandatory.

**Gate 1 (host only):** a pure deterministic adapter/score fixture must reject the overlap inversion above, duplicate kernel totals, stale event generations, mixed clock domains, missing graph UID, unsupported P2P, capacity overflow, mixed prefill/decode/MTP populations, and repeated evidence. It must preserve original raw provenance and explain why a candidate is `INELIGIBLE`, `UNKNOWN` or `SCORED`. Use exact source SHA and trace identity; do not copy QFP46's event/DAG implementation.

**Gate 2 (hold-out):** compare against at least five independent qualified placement pairs spanning two workload regimes, without using the held-out winner's wall-time to build its prediction. Require >=80% pairwise ranking accuracy on **decisive** (non-overlapping confidence) outcomes and zero forbidden topology/capacity/correctness recommendations. A tie or unknown is reported explicitly, not counted as a correct prediction. Existing qualified ABBA receipts are controls; do not queue competing hardware lanes.

**Gate 3 (handoff/closure):** for any recommended change, record its *measured exposed critical-path fraction* `f`. The theoretical maximum throughput improvement from eliminating that component is `f/(1-f)`; if `f < 0.03/1.03 ≈ 2.9126%`, a >=3% throughput win is impossible and the candidate closes without implementation. A winning offline model only advises existing flags/owner experiments. If timeline coverage or predictive accuracy fails, narrow/retire RPL01 rather than introducing a predictive scheduler. The existing >=5% two-regime actuation gate remains unchanged.

**Upstream/fork mechanism decisions:** llama.cpp's scheduler synchronizes on blocking-copy fallback, so it is the **correctness/causal baseline**, not a new optimizer to port. SGLang `python/sglang/srt/disaggregation/prefill.py` records a forward-stream completion event before overlapping an early KV transfer; vLLM `vllm/v1/core/sched/scheduler.py` defers KV block freeing when overlapping batches can still write. These illustrate event-owned lifetime and non-additive overlap; neither offers a qualified no-P2P RDNA placement speedup or a drop-in BigCherry scorer. Do not transplant their disaggregated serving architectures.

References: https://github.com/ggml-org/llama.cpp/blob/b11474/ggml/src/ggml-backend.cpp ; https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-backend.cpp ; https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/disaggregation/prefill.py ; https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py ; `QFP46`, `BCOP30`, `BCOP118`.
