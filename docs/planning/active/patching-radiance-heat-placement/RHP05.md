---
id: RHP05
order: 5
plan: patching-radiance-heat-placement
state: pending
created-at: '2026-10-15T01:35:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Request-count or timed heat reevaluation with restart-safe ownership activation

## Description

Add configurable periodic **evaluation**, not periodic mandatory movement. Use the one Radiance heat feed and ownership shadow planner to propose a candidate after 1,000 completed requests or 30 minutes (experimental starting values; benchmark/adjust). Initially activate only by **controlled restart** of engine + helpers, not by changing a live owner bitmap. Preserve an auditable last-known-good generation and rollback.

## Steps

1. Use Radiance's existing completed-request counter and monotonic time to trigger nonblocking planning on a bounded CPU control task. Record the reason (count/time/manual), collect window coverage and snapshot age, debounce concurrent triggers and avoid work on dispatch/HTTP/GPU lanes. No background work when feature disabled.
2. Evaluate RHP04 predicted gain, calibration confidence and transfer cost against configured threshold (initial hypothesis >=5% uplift), cooldown (initial 30 minutes), rate/memory limits, p95 regressions and quality evidence. If no sufficient gain, write a status counter but do not replace expert weights.
3. Persist a *proposed* ownership generation using RHP03's atomic snapshot; validate model/engine/patch/plugin digests, architecture, actual helper VRAM capacity, runtime protection map, MTP geometry, complete expert coverage and source weights before publishing accepted intent.
4. Implement controlled deployment sequence: stop admitting new requests; finish/drain in-flight requests and all mailbox lane sequences; gracefully stop helpers/engine; start helpers loading the proposed disjoint expert sets, wait for ready + runtime layer-map confirmation; start engine after eligibility/skip-owned replan; check the complete active map and startup health. **Never** flip an existing process's bitmap without loading weights.
5. Canary with small correctness and performance control; if startup, hook-map, KL, timeout, memory, GPU health or performance check fails, restart from prior validated map; if rollback fails, stop serving/fail closed and alert. Document operator override/manual pin and backoff after rollback.
6. Validate a crash at every checkpoint (before write, between helper starts, during load, between ready/engine start, after health) and concurrent request trigger/time trigger. Use existing BigCherry job/control-plane lifecycle if operating under managed deployment; do not create a second daemon.

## Detailed Solution & Technical Design

Startup ownership is set by `r3_moe_helper.cpp::select_owned/publish_owned` and `rad0002::rad_moe_remote_skip_unit` influences **initial planner**, so safe restart is the minimal supported activation. The request count is from existing service metrics; the interval clock only requests a decision. Keep phase scores and safe release of the old generation. A 'promote to remote' requires helper weight residency and engine map consistency *before* new routing. A 'promote to local' requires the engine to allocate/load weights before remote ownership ends. Restart satisfies both without runtime two-owner ambiguity.

## Files

`libr3moe` owner controller / helper startup selection, `engines/radiance/patches/rad0002_moe_remote_handoff/` or subsequent explicit patch only if needed; `tools/lab/radiance/` run orchestration; component settings and engine-specific recipes **only deviations from defaults**; RHP03 snapshot/status API; validation tests.

## Validation

Offline fake-server tests prove no active-map change while request running; 1,000-request, time-only, simultaneous triggers; stale/untrusted snapshot, map mismatch, helper death, clean and unclean restart, prior-good rollback. Brutus: multi-hour mixed workload + short controlled window, verify all generation transitions, forced rollback, KLD/greedy baseline, p95/p99 latency, completed requests, no stranded GPU slots/SHM.

## Effort & Risk

High: controlled restart entails a maintenance interruption; don't label it seamless. If external orchestration cannot safely drain/restart production, keep candidate staged for explicit operator apply.

## Standards

- Source of truth: pinned standalone Radiance 1.3.0 `89cee7ce414486c717aa2085c99e1f522e4c0ea6`; BigCherry `engines/radiance/`. Engine edits MUST be packaged as `engines/radiance/patches/radNNNN_*/{patch.py,patch.toml,SUMMARY.md}` with code-line anchors (patcher strips comments), tests with pinned blobs; helper policy lives in `libr3moe`. Re-check patch IDs before allocation.
- Follow `engines/radiance/recipes.toml` and `components/libr3/component.toml`; runtime profiles set ONLY non-defaults. Use existing BigCherry quality/ABBA/campaign/evidence facilities; no parallel scheduler, telemetry backend, loader, or compatibility alias. Disabled behaviour byte/quality/performance equivalent.
- Related active work: `RAD02` (R9700 trunk + 2 XTX helpers), `RAD03` (GPU mailbox), `RAD06` (prefill MoE), `RAD09` (merge/reconcile helper and rad0002). Integration depends on verified `RAD09` baseline. Do not overwrite another agent's branch or modify a shared patch without merging latest main.

## Acceptance Criteria

- Trigger = request threshold OR schedule; never mandatory periodic weight movement. Starting knobs experimental/default off.
- No in-flight request sees an incomplete owner map; every helper acknowledges selected generation before serving.
- Restart and crash recovery reliably restore previous proven configuration without silently losing experts or memory; failure stops rather than serving corrupt output.
- Measured speed improvement is significant and quality within established model-specific tolerance; negative/neutral result reverts and cooldowns.
- Existing Radiance metrics/state show trigger, active generation, candidate decision and rollback; no duplicate monitoring/storage.

## Notes

Depends on RHP04. This phase deliberately avoids live migration; request-count and timer schedules do NOT require a ChatGPT task or an external scheduler.
