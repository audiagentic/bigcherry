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

Startup ownership is set by `r3_moe_worker.cpp::select_owned/publish_owned` and `rad0002::rad_moe_remote_skip_unit` influences **initial planner**, so safe restart is the minimal supported activation. The request count is from existing service metrics; the interval clock only requests a decision. Keep phase scores and safe release of the old generation. A 'promote to remote' requires helper weight residency and engine map consistency *before* new routing. A 'promote to local' requires the engine to allocate/load weights before remote ownership ends. Restart satisfies both without runtime two-owner ambiguity.

## Files

`libr3moe` owner controller / helper startup selection, `engines/radiance/patches/rad0002_moe_remote_handoff/` or subsequent explicit patch only if needed; `tools/lab/radiance/` run orchestration; component settings and engine-specific recipes **only deviations from defaults**; RHP03 snapshot/status API; validation tests.

## Validation

Offline fake-server tests prove no active-map change while request running; 1,000-request, time-only, simultaneous triggers; stale/untrusted snapshot, map mismatch, helper death, clean and unclean restart, prior-good rollback. dev: multi-hour mixed workload + short controlled window, verify all generation transitions, forced rollback, KLD/greedy baseline, p95/p99 latency, completed requests, no stranded GPU slots/SHM.

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

## Implementation-grade notes (2026-10-15)

### Control state machine (restart rollout)

```text
DISABLED
  -> COLLECTING
  -> EVALUATING       (request count OR monotonic-time trigger)
  -> NO_CHANGE        (insufficient coverage/gain/cooldown; remain COLLECTING)
  -> PROPOSED         (valid candidate, model/cohort/source checked)
  -> STAGED           (checksummed full owner map, previous-good pinned)
  -> DRAINING         (stop admission; bounded request finish/cancel policy)
  -> RESTARTING       (stop engine/old helpers; load new helper weights)
  -> READY            (all helper owner maps/weights and engine slot map verified)
  -> CANARY           (quality + health + latency gate)
  -> ACTIVE           (generation durably committed)
  -> ROLLING_BACK     (failure; restart previous generation)
  -> SAFE_STOP        (rollback impossible)
```

No state transition is a token-routing operation. No candidate may enter `RESTARTING` without operator/deployment authority; in environments without an approved graceful restart surface, stop at `STAGED` for a manual apply. The trigger defaults are *experimental* and must be explicit non-default profile values; 1,000 completed requests or 1,800 monotonic seconds **since last completed evaluation**, not every raw request or wall-clock timestamp. A request arriving during in-progress evaluation must not enqueue another one. Backoff/cooldown after errors; preserve exactly one pending proposal.

### Restart transaction and journal

Write immutable proposal with digest, all owners, source count/coverage, budget proof and expected before/after benefit. Validate original active generation and engine model build before applying. Stop admission, wait on in-flight model steps (including MTP verify), wait exact GPU mailbox request/completion generations, then stop old helpers and engine. Start helpers with proposed map and confirm **both weights loaded and ownership map ready**. Start engine with the same map before the plan materialises and compare `runtime_slotmap_digest`; do not reuse stale shm or old bitmap. Canary: exact helper health, actual model generation, latency under known control, KL threshold against per-model established floor and GPU memory. If any check fails, shut down candidate and restart pinned last-good. Never serve a half-initialised allocation.

### Trigger and performance thresholds

Request and clock windows must share a single last-evaluation checkpoint. Proposed initial `min_coverage=0.99`, `min_observed_requests=1000`, `min_predicted_gain=0.05`, `cooldown_seconds=1800`, `max_owner_changes_per_apply` conservatively bounded; treat numbers as hypotheses. Long enough stale evidence may be used for ranking but not for an unqualified confidence claim; report age in steps/seconds separately from dropped fraction. Require live A/B validation and reversible rollout before unattended application.

### Crash/recovery table

| Crash point | Recovery |
|---|---|
| Before proposal persisted | Previous generation only |
| Temp snapshot during write | Ignore unverified temp; previous remains |
| After STAGED, before admission pause | Resume serving old generation |
| During drain | Wait/cancel under documented bound; no new admission |
| During helper startup | Kill candidate group; restart last-good |
| Candidate online, before CANARY acceptance | Revert on restart; no active marker advanced |
| After committed ACTIVE | New generation retained; last-good rollback pointer kept |
| Snapshot checksum/cohort mismatch | Refuse apply and serve verified fallback only |

Mock `Trigger`, `Epoch`, `snapshot/read_snapshot` exercise the core guards; they do not implement a server control plane or prove restart ordering.

## Revised 2026-10-31 (GPT review of the plan against radiance 1.14.0 and RAD23; source claims checked)

Evaluate occasionally (completed requests or monotonic time) and activate only an approved whole-map proposal through a controlled restart. Never mutate ownership in running engine/helper processes. Deterministic mode freezes ownership.

Steps:
1. Default off. Experimental thresholds 1,000 completed requests or 1,800 s, cooldown 30 min, gain >= 5% (starting guesses).
2. Evaluate in a bounded CPU task with coverage, gain, cooldown, capacity and quality gates.
3. Persist the canonical proposal and the previous-good generation; stop admission, drain ranks and lanes, restart, reload all weights, verify every participant.
4. Canary with numerical and performance evidence; roll back by restart; fail closed when a complete map cannot be restored. A helper that held uniquely owned experts disappearing is a restart, not a silent reassignment.

Validation: Trigger deduplication, crash during load, map mismatch, partial restart, rejected proposal, forced rollback, deterministic mode, MTP drain, no ownership change during an active request.

## Stage A by hand, first result (2026-11-01)

`tools/lab/radiance/heat-placement/heat-drift.py` (#489) checks the active `--heat` file against a recorded histogram: the share
of live routes the heads and the RX 6900 XT tail take under it, against a file re-ranked from the live counts. On the recorded
histograms the tail takes 1.4-1.6% of routes with matched heat, 1.9-2.4% with the mixed default, 4.3-8.3% mismatched and up
to 12.2% with the old static file; it costs ~10x a head a route. One loop on flashnext-4gpu: record during the 42K probe ->
heat-drift (tail 6.85% against 1.72% re-ranked, advise) -> restart with the written file -> the same probe 14.31 -> 12.72 s
(-11%), the cached 14K follow-up 1.01 -> 0.75 s, the engine's wait on the helpers 6.06 -> 4.22 s and its p95 a layer 13.7 ->
4.6 ms (the tail 4.2 -> 1.6 ms a request). In-sample (recorded and replayed on the same text): the held-out case is the next
measurement before any automatic trigger.

## Held-out (2026-11-01, `runs/rhp05-holdout`)

The same 42K probe on the other half of the documentation (set B: the doc list read from the end), flashnext-4gpu:

| heat file | 42K request | engine wait | tail a request |
|---|---|---|---|
| mixed default | 12.75 s | 3.92 s | 1.63 ms |
| recorded on set A (the in-sample file above) | 13.35 s | 3.91 s | 1.81 ms |

Heat recorded on one text did not carry to another of the same kind: the set-A file is 5% slower on set B than the mixed
default, which already puts only 1.46% of set B's routes on the tail (re-ranked on set B's own heat: 0.34%), so the tail
finished before the heads (1.6 against 3.7 ms a request) and a restart had nothing to win. heat-drift.py now keeps the file
while the tail's share is under --min-share (2.5%, #489); it still advises the in-sample case (6.85%). The trigger for stage
A is therefore sustained drift of the served workload's own heat past that floor, not a single recording.

