---
id: RHP06
order: 6
plan: patching-radiance-heat-placement
state: pending
created-at: '2026-10-15T01:35:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: L
---

# Optional live cross-GPU expert migration with versioned ownership epochs

## Description

Only after RHP05 proves workload drift frequently justifies reallocation, implement a GPU-safe live ownership change while requests continue. Radiance's local `Mover` already knows asynchronous transfers and quarantine; adapt its **invariants** to cross-process helper weights, without pretending a helper is a normal same-process `Tier`. No hot-swap without complete staged weights and generation fencing.

## Steps

1. First establish an Amdahl/economics gate: observed stale placement waste, number of changes per hour, bytes per expert, interruption cost of RHP05 restart, expected recoup time and peak auxiliary VRAM. If restart meets objectives, close this item as **not justified**, not partially implement.
2. Design owner-map epochs across engine + both helpers: `(layer,expert)` global IDs, ownership array, per-GPU residency-ready bit, 64-bit generation, hash and ACK. Mutable topology/capacity only between validated epochs. Map protects layers with BF16/protected slots and MTP; `R3_MOE_HELPER_SKIP_OWNED` currently maps weights away during planner construction, requiring explicit runtime allocation/loading or a bounded local fallback source.
3. Stage new owner weights into pre-reserved VRAM/shadow slots; stream asynchronously using helper loader and Radiance mover semantics. Respect GPU budget, host/SSD staging, source availability, no P2P, mapped-host GPU mailbox ordering and helper lane isolation. Preserve old owner weights until last old-epoch operation retires.
4. At a **global safe step boundary**, quiesce or fence all ranks/helpers, drain old mailbox requests and per-lane sequences, confirm `ready(epoch+1)` from all participants, atomically publish the new map and the corresponding device-side bitmap/table. No mixed epochs inside one forward; use epoch in request/completion and a strict failure path. CPU decides; GPU continues GPU-side execution and waits as required at barrier, never CPU-spin relay of tokens.
5. Add rollback and fault injection for helper crash, stalled mapped host writes, mismatched checksum, partial stage, memory pressure, old epoch still in flight, generation wrap and quant/output mismatch. Revert to last good owner map only if all old weights still resident; otherwise stop serving safely.
6. Measure amortised throughput and tail latency versus RHP05 controlled-restart strategy over changing workload phases; measure staging bytes/latency, GPU waits, memory headroom, numerical noise, version mismatch and quality. Promote only if end-to-end gain remains under realistic change frequency.

## Detailed Solution & Technical Design

The RHP05 static ownership maps are built before load and `rad0002` can mark remote-owned experts as `Tier::Mapped` never-read placeholders. Live migration requires new resident weights / reversible host sources at the destination as well as a versioned **execution site** selection. Reuse `Mover`'s shadow slot, quarantine, event and publication rules where possible; do not call its local `HeatEngine::set_tier` for a remote owner assignment. The `moe_remote_shm.h` protocol v11 currently has exact generation request completion but no **placement epoch**; extend protocol with explicit negotiation/version bump and plugin tests if gated into implementation. Enforce old/new map lifetimes across batch+MTP speculative passes and graph replay.

## Files

`engines/radiance/patches/rad0002_moe_remote_handoff/` (or a new narrowly scoped engine patch), `engines/radiance/components/libr3/helper/{moe_remote_shm.h,moe_remote_hook.cpp,moe_remote_gpu.h,r3_moe_helper.cpp}`; Radiance pinned `core/place/mover.{h,cpp}` ABI adaptation if justified; component settings, patch package tests and GPU stress harness.

## Validation

Deterministic simulation of every state transition and crash point; GPU poison/guards, memory-safety and handshake fuzz; 1/8/64/4096-token graph replay, two-helper races, long prefill under enqueue run-ahead, mixed MTP. No missing or duplicate expert contribution, incorrect TP slice or stale pointer; KLD/greedy within established floor; workload-shift A/B with detailed pause/wait and memory counters.

## Effort & Risk

Very high: correctness, deadlocks and missing mapped weights. Implement as a separate opt-in capability; do not block production RHP05 while pursuing it.

## Standards

- Source of truth: pinned standalone Radiance 1.3.0 `89cee7ce414486c717aa2085c99e1f522e4c0ea6`; BigCherry `engines/radiance/`. Engine edits MUST be packaged as `engines/radiance/patches/radNNNN_*/{patch.py,patch.toml,SUMMARY.md}` with code-line anchors (patcher strips comments), tests with pinned blobs; helper policy lives in `libr3moe`. Re-check patch IDs before allocation.
- Follow `engines/radiance/recipes.toml` and `components/libr3/component.toml`; runtime profiles set ONLY non-defaults. Use existing BigCherry quality/ABBA/campaign/evidence facilities; no parallel scheduler, telemetry backend, loader, or compatibility alias. Disabled behaviour byte/quality/performance equivalent.
- Related active work: `RAD02` (R9700 trunk + 2 XTX helpers), `RAD03` (GPU mailbox), `RAD06` (prefill MoE), `RAD09` (merge/reconcile helper and rad0002). Integration depends on verified `RAD09` baseline. Do not overwrite another agent's branch or modify a shared patch without merging latest main.

## Acceptance Criteria

- Documented measured economics justify live over restart; otherwise shelve.
- Complete epoch fencing, stable retirement of old owner weights, safe rollback and no in-flight mixed owner maps.
- Live A/B beats RHP05 on representative phase-shifted service after accounting for transfers/extra VRAM and p95/p99 latency; no quality regression or helper deadlock.
- No new route counter, external worker service or redundant observability; same RHP02 heat and RHP03 snapshots/metrics.

## Notes

Depends on RHP05; strictly optional. Does not change the model router or numerical method.
