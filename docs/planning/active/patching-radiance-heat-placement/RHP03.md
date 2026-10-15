---
id: RHP03
order: 3
plan: patching-radiance-heat-placement
state: pending
created-at: '2026-10-15T01:35:00+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# Extend Radiance telemetry and persist one placement snapshot, not a heat database

## Description

Expose the shared heat/ownership decision state through Radiance's existing expert-plane/status and `/metrics` paths; persist only small, versioned ownership snapshots needed for controlled restarts and comparison. Do **not** build a new Prometheus exporter, polling service, SQLite heat store, artifact framework, or competing observability namespace.

## Steps

1. Inventory upstream `core/engine.cpp` expert-plane (`ExpertPlane`), `core/server/metrics.{h,cpp}`, `HeatStats`, `MoverStats` and BigCherry existing `artifacts/` + `docs/evidence/` conventions. Reuse stable counters and no per-expert Prometheus labels.
2. Surface cardinality-bounded aggregates: observed/credited/dropped/pending histograms and age by prefill/decode, total routes and remote share, local VRAM hit, helper requests/latency, placement generation, last evaluation/apply/rollback, moved bytes, GPU ownership byte allocation, predicted vs measured uplift, churn/readmits.
3. Extend expert-plane status with **GPU owner** separate from **tier**; do not overload the existing tier enum with XTX0/XTX1. Return a consistent generation snapshot rather than half-applied data. For deterministic/off mode mark adaptation disabled, not '0 experts'.
4. Define compact, versioned machine-local `<work>/state/radiance/<model-digest>/<hardware-cohort>/placement.json`: schema, model/container hash, engine/plugin/patch digest, immutable GPU cohort/slot geometry, source heat-window coverage, selected owner per eligible (layer,expert), digest, creation/apply timestamps, previous good generation, decision reason. Atomic write/fsync+rename and checksum verification, stale file rejection.
5. Raw routing counts/temporary profiles remain in RAM or experiment-specific `artifacts/<run-id>/`; evidence summaries in `docs/evidence/<run-id>/` after validation. Avoid recording prompts, user data or an unbounded per-request event log. Document retention/rotation and recovery after truncated/corrupt snapshot.

## Detailed Solution & Technical Design

Do not store the online cache's entire moving float heat vector just to make placement auditable; a bounded optional heat summary attached to the generation suffices. Machine-local state must not be a source of truth for current in-GPU residency; Radiance mover already owns it. Snapshot describes GPU **ownership** only and is applied after verifying real weights are loaded and map agreement on every rank/helper. Observability reads cannot stall serving or force GPU sync.

## Files

Radiance patch package for existing `core/server`/engine observability when necessary; `libr3moe` small status interface; `tools/lab/radiance/` reporting; `docs/reference/engines/` relevant runtime doc or model page after RAD07; `tools/tests/radiance/` and patch tests.

## Validation

Snapshot round-trip, checksum, wrong model/host/device order, partial write, stale generation, permissions, deterministic/off. Metrics scrape repeated during load, shutdown and prospective epoch change; cardinality bounded with two helpers; compare aggregate counts to HeatEngine/transform traces; no additional critical-path device sync.

## Effort & Risk

Risk: readings from several ranks may double count TP-shared routes or expose inconsistent half-generation state; document aggregation dimensions.

## Standards

- Source of truth: pinned standalone Radiance 1.3.0 `89cee7ce414486c717aa2085c99e1f522e4c0ea6`; BigCherry `engines/radiance/`. Engine edits MUST be packaged as `engines/radiance/patches/radNNNN_*/{patch.py,patch.toml,SUMMARY.md}` with code-line anchors (patcher strips comments), tests with pinned blobs; helper policy lives in `libr3moe`. Re-check patch IDs before allocation.
- Follow `engines/radiance/recipes.toml` and `components/libr3/component.toml`; runtime profiles set ONLY non-defaults. Use existing BigCherry quality/ABBA/campaign/evidence facilities; no parallel scheduler, telemetry backend, loader, or compatibility alias. Disabled behaviour byte/quality/performance equivalent.
- Related active work: `RAD02` (R9700 trunk + 2 XTX helpers), `RAD03` (GPU mailbox), `RAD06` (prefill MoE), `RAD09` (merge/reconcile helper and rad0002). Integration depends on verified `RAD09` baseline. Do not overwrite another agent's branch or modify a shared patch without merging latest main.

## Acceptance Criteria

- One authoritative telemetry surface and one small machine-local ownership snapshot; no new DB/event service.
- Restart rejects mismatched/corrupt placement and safely loads last validated generation or static fallback.
- Captured lag/drop counters and time/phase coverage visible; raw per-expert IDs absent from metrics label sets.
- Disabled path identical; snapshots only when configured; no polling overhead proportional to number of requests.

## Notes

Depends on RHP02. Reuses existing BigCherry artifacts and experiment evidence lifecycle.
