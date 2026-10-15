---
id: RHP02
order: 2
plan: patching-radiance-heat-placement
state: pending
created-at: '2026-10-15T01:35:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: L
---

# Single Radiance heat observation feed to libr3moe; remove duplicate route collection

## Description

Expose a **read-only, one-to-many** observation from Radiance's existing histogram consumption, not a parallel GPU-side aggregation. Feed both the current `HeatEngine` and a separate cross-GPU ownership *policy consumer* from the same source events. Remove production dependence on `R3_MOE_HELPER_HISTOGRAM`/`R3_MOE_HELPER_HEAT` only after matching coverage and ranking; avoid compatibility aliases.

## Steps

1. After RHP01, define a narrow versioned `abi/rad_moe_heat.h` **optional hook** in a new Radiance patch package (or extend rad0002 if consistent with ownership ABI): immutable {model-global layer/expert domain, rank, step/epoch, phase or step-kind, counts, count length, multiplicity, completeness/overrun markers}; document buffer lifetime (valid during callback only). No pointers or blocking across processes.
2. Publish from the existing CPU `tiering_dispatch` completed-bank loop exactly once per credited histogram; do not dispatch callbacks from `rad_route_report`, alter router math, copy counts again from GPU, or wait for future samples. Keep the existing `HeatEngine::observe` order and `last_histogram_reports`.
3. Make observation work if `HeatEngine::enabled()==false` (all resident or helper-owned units `Tier::Mapped`) without turning **placement** on; `--deterministic` suppresses adaptation but may permit explicit read-only diagnostics. Avoid observing twice if multiple ranks share equivalent global counts; define rank selection/aggregation and MTP protected-layer semantics.
4. Keep callback constant-time and bounded: queue or accumulate small immutable host snapshots out of the inference dispatch loop; no filesystem I/O, sorting, mutex contention with helpers, RPC, CPU callback on GPU critical stream or extra copy. Document queue full/restart policy and coverage counters. Consumer may use the same score formula or phase-specific window, but must not fork the *collection path*.
5. Wire `libr3moe` ownership policy to the shared observations, compare its (layer,expert) counts and ordering against current `R3_MOE_HELPER_HISTOGRAM` diagnostic on matched runs; test prefill/decode split and protected slots. Once parity is accepted, remove duplicated GPU allocations, route counting kernel and heat file env vars from production component/settings/tests/docs in one migration, with no deprecated aliases.

## Detailed Solution & Technical Design

`HeatEngine` is a policy over *movable local units*, not a universal GPU-owner model. Do not force it to treat remotely owned experts as locally resident or invent movable mapped units. The shared **input histogram** is the invariant; `libr3moe` may maintain a bounded CPU score/window for remote ownership and reuse upstream's decay/hysteresis algorithm rather than a duplicate device telemetry stack. Ensure `rad0002` map/owner hooks still refer to exact runtime slot IDs; protected experts are never remapped merely from container IDs.

## Files

New/extended package in `engines/radiance/patches/` touching upstream `abi/`, `core/engine.cpp` (and `ctx` only when required); `engines/radiance/components/libr3/helper/{moe_remote_hook.cpp,moe_remote_gpu.h,r3_moe_helper.cpp}`; `component.toml`, patch tests, module parity tests and docs.

## Validation

Card-free callback lifetime, version mismatch, no callback, disabled-static mode, double-credit, phase attribution, MTP multiplicity and partial coverage tests. Hardware shadow comparison against existing histogram with same consumed step set, including 32K prefill and two helpers; check no CPU critical-path stalls and unchanged KL/greedy outputs.

## Effort & Risk

High: long-prefill host run-ahead and remote-only mapped units can hide holes in the normal HeatEngine path. Keep a diagnostic comparison gate but no permanent second collector.

## Standards

- Source of truth: pinned standalone Radiance 1.3.0 `89cee7ce414486c717aa2085c99e1f522e4c0ea6`; BigCherry `engines/radiance/`. Engine edits MUST be packaged as `engines/radiance/patches/radNNNN_*/{patch.py,patch.toml,SUMMARY.md}` with code-line anchors (patcher strips comments), tests with pinned blobs; helper policy lives in `libr3moe`. Re-check patch IDs before allocation.
- Follow `engines/radiance/recipes.toml` and `components/libr3/component.toml`; runtime profiles set ONLY non-defaults. Use existing BigCherry quality/ABBA/campaign/evidence facilities; no parallel scheduler, telemetry backend, loader, or compatibility alias. Disabled behaviour byte/quality/performance equivalent.
- Related active work: `RAD02` (R9700 trunk + 2 XTX helpers), `RAD03` (GPU mailbox), `RAD06` (prefill MoE), `RAD09` (merge/reconcile helper and rad0002). Integration depends on verified `RAD09` baseline. Do not overwrite another agent's branch or modify a shared patch without merging latest main.

## Acceptance Criteria

- Exactly one authoritative GPU routing histogram producer and existing Radiance host bank transfer; both consumer policies see the same credited events.
- Old diagnostic vs new feed agrees **exactly** on matched (rank,step,layer,expert) counters, including MTP weighting, with full explicit coverage report. No silent expert-ID remapping.
- Default-off path unchanged; remote helper still reproduces established quality/speed baseline; no new serving GPU sync or duplicate GPU heat allocation in production.
- Existing duplicated collection/storage variables removed when parity passes, no legacy path retained.

## Notes

Depends on RHP01 and merged/verified RAD09. Does not implement ownership migration; that is RHP05/06.
