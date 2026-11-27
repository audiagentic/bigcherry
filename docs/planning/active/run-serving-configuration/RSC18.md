---
id: RSC18
order: 0
plan: run-serving-configuration
state: pending
created-at: '2026-11-27T03:42:08.275428+00:00'
breadth: ''
skill: advanced
created-by: agent
work: L
priority: P1
---

# Shared KV live accounting, mixed-length admission and scheduler progress

## Description

RSC16 stage C owner, depends on RSC17 capacity contract. Reuse KVManager and Scheduler; code guide E-F. No second scheduler, cache allocator or per-client maximum reservation.

## Steps

1. Run existing 2.3 mem/sched/paging-fence fixtures unchanged with RSC17 geometry; reuse blocks_still_needed/refill_cap, can_grow window logic, ensure_admission and native external-capacity/error states. 2. Add only missing exact page-demand, multi-group failure/retry and progress cases; fix demonstrated mismatch, preserve legitimate safe recycling/recall. 3. Reuse existing occupancy/availability/cache-only counters and adapter; keep legacy metrics, expose actual backed vs recallable. 4. Measure rectangular overhead only; hand off integrated mixed-length qualification to RSC16.

## Detailed Solution & Technical Design

Code guide E-F, H and I4: audit existing scheduler first, reuse demand helpers and consolidate only demonstrated differences. Distinguish usable free, recallable lent, held and cache-only reclaimable pages; state slots separate. Retain existing governor and bounded progress.

## Code Samples & Guidance

core/mem/kv.cpp blocks_for/can_grow/ensure/available_blocks; core/sched/scheduler.cpp fresh_prefill_can_fit/preempted_can_prefill/metrics; metrics.h and core/server adapters.h, iface.h, dashboard.cpp, metrics.cpp, history.cpp. Guide I4 gives exact conservation identities and metric compatibility rules.

## Files



## Validation

Existing mem/kv_window/sched/checkpoint/server and engine e2e tests; shared prefixes, mixed lengths, exhaustion, speculative rollback, deferred release, bounded preemption, no lost state. Compare actual committed/used/free/evictable page ledgers, including error injection.

Guide I4: deterministic page conservation (total=held+usable free+lent), failed recall, flex room, shared prefix/COW, window wrap, state slots, deferred release, zero/sentinel, multi-group capacity and exact metric scope; no hot-path allocation/scan regression.

Test 65+1 tokens with B16 = 6 pages; distinguish active/queued/idle-retained, cold L alone versus busy pool; verify pressure and impossible fit separately; actual group page receipts under mixed requests.

Guide L4: existing fixture plus shared-only gate resolution, generation/embed context boundaries, mixed-page and COW/progress tests, allocator error during RUNNING with other request/cache vs ordinary FULL and retry; no unnecessary eviction, no lost readable keys, retained refs/fences correct.

## Effort & Risk

No new scheduler, allocator or telemetry stream. Ragged tables/dynamic state only if measured dominant, with separate ABI/lifetime review.

## Standards



## Acceptance Criteria

No max-window reservation per client; existing length/slot guards; exact per-group page accounting and safe failure; progress under contention, no live eviction/truncation; unchanged legacy path; runtime metrics reflect actual capacity. RSC16 owns GPU mixed-length and cross-family release acceptance.

Existing and new metrics have documented semantics; distinguish immediately usable, recallable and cache-only reclaimable; no summed logical capacity across ranks/groups; runtime tests and performance baseline preserve progress and safety.

## Notes

RSC16 second design review/code guide H: existing Scheduler::init/add_locked/fresh_prefill_can_fit already use actual context/prompt demand. Audit first, make only proven corrections. can_grow vs ensure runtime-window geometry diverges; verify or share read-only demand helper, preserve COW and elastic semantics. Explicit context overflow vs transient pressure, no silent truncation in new contract. Keep rad0036/rad0028 dependencies.

Consolidated cleanup I4: metrics and demand accounting belong here; RHP16/20 live placement remain separate.

Fourth review guide J2/J4: explicit minimum is packed capacity, not arbitrary mixed workload guarantee. Account actual sum of per-session rounded pages, sharing/COW and progress; preserve legacy flex and distinguish backed vs reserved, actual recallable. Reuse existing cache_only_blocks counters.

Guide K2/K5/K6 replaces old assumption can_grow lacks window cap: 2.3 adds native cap and runtime window geometry; verify real keep_from versus ensure. Pass 0 recycles and pass 1 recalls, so do not require global transaction rollback if safety/retry proven. Native Fleet wait and transfer fences must survive. No second scheduler, demand gate, metric stream or blanket ABI redesign.

Readiness review guide L4: native prefill_room_gate defaults false; resolve on only in shared mode, reject explicit off. Reuse scheduler; runtime reference-plus-ring is not a bounded lookback window and is deferred in shared mode. Validate generation L-1 plus 1, L prompt refusal, exact-L embeddings, speculation guard; running ensure_from currently drops actual status (distinguish FULL from other errors, preserve native fatal/fence policy). No unconditional rollback or alternate scheduler.

RV4754: full new-mode admission/metrics follows RSC17 after baseline. In bump preserve existing actual-demand growth, prefix-aware governor and failure/progress/rollback tests; fix proven legacy defects only. Do not introduce unconditional room-gate or new L/S/C mode.

## Change Log

- 2026-11-27T03:42:08.275428+00:00 (created-by): Created by agent
- 2026-11-27T03:47:57.369602+00:00 (updated-by): Updated: section:notes, section:steps
- 2026-11-27T03:49:13.270930+00:00 (updated-by): Updated: section:notes
- 2026-11-27T04:00:52.358563+00:00 (updated-by): Updated: section:detailed_solution, section:code_samples, section:effort_risk, section:notes
- 2026-11-27T04:01:25.833107+00:00 (updated-by): Updated: section:steps, section:validation, section:acceptance_criteria
- 2026-11-27T04:18:47.331154+00:00 (updated-by): Updated: section:notes, section:validation
- 2026-11-27T07:13:21.540426+00:00 (updated-by): Updated: section:steps, section:notes
- 2026-11-27T07:29:29.417425+00:00 (updated-by): Updated: section:notes, section:validation
- 2026-11-27T07:30:36.115403+00:00 (updated-by): Updated: section:notes
- 2026-11-27T15:12:50.065161+00:00 (updated-by): Updated: section:notes
