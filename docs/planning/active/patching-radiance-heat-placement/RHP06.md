---
id: RHP06
order: 11
plan: patching-radiance-heat-placement
state: pending
created-at: '2026-10-15T01:35:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: L
---

# Stage 1: live helper expert migration - slot manager, async host->VRAM upload, retirement

## Description

Stage 1 with RHP11 (no longer optional - owner direction 2026-11-01). Each helper manages its own physical expert slots and moves an expert in by uploading it from the immutable host weight source (no P2P on this host), never from the old owner's GPU. Applies the RHP04 stage 1 policy through RHP11's epochs. Within stage 1 the engine-local / remote split stays fixed; any helper-owned expert may move to any eligible helper (including one that owns none today) if it fits.

## Steps

1. First establish an Amdahl/economics gate: observed stale placement waste, number of changes per hour, bytes per expert, interruption cost of RHP05 restart, expected recoup time and peak auxiliary VRAM. If restart meets objectives, close this item as **not justified**, not partially implement.
2. Design owner-map epochs across engine + both helpers: `(layer,expert)` global IDs, ownership array, per-GPU residency-ready bit, 64-bit generation, hash and ACK. Mutable topology/capacity only between validated epochs. Map protects layers with BF16/protected slots and MTP; `R3_MOE_WORKER_SKIP_OWNED` currently maps weights away during planner construction, requiring explicit runtime allocation/loading or a bounded local fallback source.
3. Stage new owner weights into pre-reserved VRAM/shadow slots; stream asynchronously using helper loader and Radiance mover semantics. Respect GPU budget, host/SSD staging, source availability, no P2P, mapped-host GPU mailbox ordering and helper lane isolation. Preserve old owner weights until last old-epoch operation retires.
4. At a **global safe step boundary**, quiesce or fence all ranks/helpers, drain old mailbox requests and per-lane sequences, confirm `ready(epoch+1)` from all participants, atomically publish the new map and the corresponding device-side bitmap/table. No mixed epochs inside one forward; use epoch in request/completion and a strict failure path. CPU decides; GPU continues GPU-side execution and waits as required at barrier, never CPU-spin relay of tokens.
5. Add rollback and fault injection for helper crash, stalled mapped host writes, mismatched checksum, partial stage, memory pressure, old epoch still in flight, generation wrap and quant/output mismatch. Revert to last good owner map only if all old weights still resident; otherwise stop serving safely.
6. Measure amortised throughput and tail latency versus RHP05 controlled-restart strategy over changing workload phases; measure staging bytes/latency, GPU waits, memory headroom, numerical noise, version mismatch and quality. Promote only if end-to-end gain remains under realistic change frequency.

## Detailed Solution & Technical Design

New helper units (e.g. worker_residency.h, worker_migration.h, worker_epoch.h). Fixed-stride or size-classed slots derived from the backend layout, each holding a whole executable expert (gate/up codes and scales, down codes and scales, metadata). Slot states FREE -> RESERVED -> COPYING -> READY -> ACTIVE -> RETIRING -> FREE with ready and retire events and last_epoch. Refactor moe_remote_gpu.h's startup upload_weight() into a reusable weight-source/layout step plus an async destination-slot upload; no pointer is published until every plane is complete. Hard budget: active + pending + retiring + scratch <= hard budget, a slot charged until reclaimed; reserve at least one incoming expert per helper; a 2-way exchange between full helpers needs staging space or a validated multi-epoch transaction, else DEFERRED_CAPACITY. Bounded pinned bounce buffers; start with one in-flight upload per destination helper and a small global byte budget; record transfer times for stage 2. A forced-move test mode uses the production path and bypasses only the economic threshold.

## Code Samples & Guidance



## Files

`engines/radiance/patches/rad0002_moe_remote_handoff/` (or a new narrowly scoped engine patch), `engines/radiance/components/libr3/worker/{moe_remote_shm.h,moe_remote_hook.cpp,moe_remote_gpu.h,r3_moe_worker.cpp}`; Radiance pinned `core/place/mover.{h,cpp}` ABI adaptation if justified; component settings, patch package tests and GPU stress harness.

## Validation

Replay: --replay-sweep with an ownership-transition schedule - replay a captured request under owner A, move the expert to B, commit, replay under B; byte-identical loaded weights; correct pointers; no uninitialised planes; every route exactly once; no stale references; old slot poisoned after retirement shows no use-after-free; no slot leak. Same-architecture moves: bit-identical per-expert contributions. XTX <-> gfx1030 moves: CPU reference / KLD (different kernels). Forced churn: >=1,000 transitions including XTX<->XTX hot experts, moves to and from the tail, 2-way exchanges near capacity, commits with pre-queued lanes, after large-lane prefill, with the CPU combine, MTP traffic, concurrent uploads, mid-copy faults and delayed retirement; zero invalid owners, missing or duplicate contributions, stale references, hangs or leaked slots; no unbounded VRAM growth. End-to-end: the RHP12 gate.

## Effort & Risk

Very high: correctness, deadlocks and missing mapped weights. Implement as a separate opt-in capability; do not block production RHP05 while pursuing it.

## Standards

- Source of truth: pinned standalone Radiance 1.3.0 `89cee7ce414486c717aa2085c99e1f522e4c0ea6`; BigCherry `engines/radiance/`. Engine edits MUST be packaged as `engines/radiance/patches/radNNNN_*/{patch.py,patch.toml,SUMMARY.md}` with code-line anchors (patcher strips comments), tests with pinned blobs; helper policy lives in `libr3moe`. Re-check patch IDs before allocation.
- Follow `engines/radiance/recipes.toml` and `components/libr3/component.toml`; runtime profiles set ONLY non-defaults. Use existing BigCherry quality/ABBA/campaign/evidence facilities; no parallel scheduler, telemetry backend, loader, or compatibility alias. Disabled behaviour byte/quality/performance equivalent.
- Related active work: `RAD02` (R9700 trunk + 2 XTX helpers), `RAD03` (GPU mailbox), `RAD06` (prefill MoE), `RAD09` (merge/reconcile helper and rad0002). Integration depends on verified `RAD09` baseline. Do not overwrite another agent's branch or modify a shared patch without merging latest main.

## Acceptance Criteria

Experts move between helpers while serving with no restart; forced-churn, replay and protocol tests pass; RHP12's 8-session, 6-stream and 42K probes show idle-framework median step-time regression <=1% (p95 <=2%) and during migration <=2% median / <=5% p95; quality gates pass for heterogeneous moves.

## Notes

Depends on RHP05; strictly optional. Does not change the model router or numerical method.

Plan source: GPT staged plan 2026-11-01 (req_main_8f893e8abe79). Prior art in radiance core/place/mover.* (within-rank shadow-slot transfers). Stage 4 extends this to engine<->helper moves.

## Implementation-grade notes (2026-10-15)

### Gate before any implementation

Measure total actual service interruption for RHP05, frequency of changed optimal placements, network/PCIe and GPU memory cost of transferring expert bytes, and expected payback. If changes are rare and planned restarts are acceptable, **close RHP06 as intentionally not implemented**. Never make live migration a prerequisite for RHP05 production use.

### Epoch and quiescence protocol

A map generation is a 64-bit monotonically increasing **placement** epoch, distinct from existing v11 mailbox **request completion** sequence. Every GPU mailbox request is bound to a generation; helper completion must contain the same epoch, exact request generation and checksum. Participants: all engine ranks and all configured helpers; every participant must ACK `(epoch+1, map digest, loaded weights, layer slot geometry)`. Old epoch remains live until no consumer retains it, including draft/verify passes and submitted graph replay operations.

Recommended phases: `ACTIVE(n)` -> `PROPOSED(n+1)` -> `LOADING(n+1)` -> `READY(n+1)` -> `QUIESCING(n)` -> `COMMITTING` -> `ACTIVE(n+1)`, with `ABORT` on any mismatch. Publish ownership only once all rank streams reach a global safe boundary and all old tickets retired. Engine/remote hook must use a single table per model step, not per route, to prevent mixed-owner outputs. Reuse mapped-host release/acquire and GPU-stream fences already in the mailbox protocol; do **not** substitute CPU polling for a GPU-lane dependency.

### Memory and weight lifetime

A remote-owned engine expert may be intentionally absent locally due to `R3_MOE_WORKER_SKIP_OWNED`; consequently switching remote -> local requires **new local weight allocation and verified full weight load**, not just new bitmap bits. Local -> remote requires the helper to preload the correct quantized whole-expert weights and keep engine copy until all prior steps finish. Per-device peak budget must include old + new residency, transfer staging, kernel scratch, KV and host pool; if it does not fit, refuse the migration or use RHP05 restart. Restore pool/slab accounting after abort, including shadow slots and in-flight events. Never repurpose protected or MTP-only weights.

### Failure matrix

Wrong generation, same-generation checksum mismatch, helper crash before ACK, helper crashes after ACK before commit, event never completes, stale DMA writes, mapped-host visibility lag, helper capacity exhaustion, GPU OOM, rank disagreement, missing local source for skipped weight, cancel during prefill, simultaneous ownership updates, rolling request-counter wrap, restart during live migration and graph replay referencing old weights. Validate exact output contribution once per selected expert; rollback may need `SAFE_STOP` if old weights were already released.

### Test design

CPU protocol mock proves ACK/quiescence/generation fencing; GPU stress requires 0,1,2 helpers, 1/8/64/1024/4096 rows, protected-slot perturbations, long prefill/MTP decode, fault injection on every transition, poison freed buffers, host-mapped visibility and TSAN/ASAN where practical. Verify `KL/top-1` vs current helper-off/control, no extra numerical divergence due to ownership and no repeat expert contributions. Record transfer bytes, overlapping work, p95/p99 service latency and negative-case recovery.

## Revised 2026-10-31 (GPT review of the plan against radiance 1.14.0 and RAD23; source claims checked)

Only if measured drift and restart interruption justify it. Not an update of the owner bitsets: a placement epoch (64-bit, distinct from the mailbox request sequence) must cover every operation of a model step including MTP passes and already-enqueued GPU work. See RHP11.

Steps:
1. Quantify restart cost, placement changes per hour, staging bytes, shadow VRAM, payback time.
2. Extend protocol v19 with a negotiated version and the epoch; stage and validate destination weights and keep the old residency until old-epoch operations retire.
3. Bind each forward/MTP step to one immutable map; quiesce all ranks and helpers at a global safe boundary; publish after (epoch, map digest, residency-ready) acknowledgements.

Validation: State-machine fault injection, epoch overlap, GPU stream and graph replay, helper death, incomplete staging, generation mismatch, memory pressure, repeated flips; compare pause time and p99 with RHP05.

## Change Log

- 2026-11-01T22:51:18.702471+00:00 (updated-by): Updated: section:title, order=11, priority='P0', section:description, section:detailed_solution, section:validation, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20261101_225732_planning-a-staged-roadmap-for_4092
- 2026-11-01T22:57:39.628248+00:00 (updated-by): Updated: section:ledger-events
- chg_20261101_233905_design-spec-for-moving-experts_5737
- 2026-11-01T23:39:13.660857+00:00 (updated-by): Updated: section:ledger-events
