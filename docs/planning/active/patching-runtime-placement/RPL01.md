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
