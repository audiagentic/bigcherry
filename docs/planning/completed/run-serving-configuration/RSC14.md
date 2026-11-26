---
id: RSC14
order: 0
plan: run-serving-configuration
state: completed
created-at: '2026-11-19T22:22:14.183742+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: S
---

# Flash-Next production start fails out of memory when builds run beside it: make host-RAM admission robust

## Description

2026-11-20: the no---set production smoke of installed bc-radiance-1170.34.0 (flashnext-4gpu, RAM n-gram table 47.7 GiB) stopped with 'E startup stopped: out of memory' while C++ builds ran on the host; the rerun minutes later (MemAvailable ~79 GiB) passed. bigcherry-2 saw the same borderline earlier (engine admission needed ~74.8 GiB, 74.13 available). The admission is correct to refuse, but a production start that fails whenever a build or page cache grows is fragile: the plan admits at owner start, and the engine's own host-room check happens later after memory moved.

## Steps

Reproduce and measure owner-versus-native preweight host-room disagreement under concurrent builds, cgroup limits, pinned-memory limits and helper startup. Reuse RSC03 typed live resource admission and RSC11 transferred RSC10 structured shortfall/refit: one retry only when native reports needed/available/shortfall before weights or outstanding copies; recompute only derived resource outputs from live facts and measured margin, otherwise refuse early. No owner prepin by default, guessed margin or second retry. Verify one forced refusal/refit and repeated contention, with exact owner receipts and stop/restore.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Reproduce: production start of flashnext-4gpu while a radiance lane build runs; it succeeds or refuses before load with a named reason; owner and engine admission agree.

## Effort & Risk



## Standards



## Acceptance Criteria

A concurrent build may cause a named early resource refusal, not a late OOM due to stale admission. Preserve RSC11's forced typed shortfall, one successful refit, double-refusal and three six-stream soak gates. No claim that memory cannot change after admission; native final allocator remains authoritative.

## Notes

2026-11-25 serving/configuration closure sweep: docs/design/radiance-serve-config-review-20261125.md. RSC11 transferred RSC10 acceptance already owns structured preweight host shortfall and one retry; RSC03 owns measured ledger. Reuse only that implementation; this item owns reproduction and contention/failure fixtures. No owner prepin by default (double RAM charge), no guessed extra margin, no separate retry/launcher. Verify actual cgroup/pin/MemAvailable, helper residency, real concurrent build and before weight loading; success or clear early refusal is acceptable, no retry after partial weights/outstanding copy. Preserve RSC11 three six-stream soaks and forced one retry evidence. Distinguish resource refusal from benchmark or launch failure.

2026-11-26 closeout: #1658 (the engine's typed host-room refusal at the health gate is a named resource refusal: owner exit 97, message says memory changed since admission, the valid stored plan is kept) and #1668 (rad0011: a plan mismatch the host cannot hold is re-read as the structured host-room refusal, exit 4, instead of drift; found by the first real contention run, which showed the engine placing the pinned n-gram table on disk and failing with BC_CFG_PLAN_MISMATCH). Hardware: the rerun on 2020.22.0 passed (first start measured, forced contention try 1: named early refusal via the engine's preweight check, owner exit 97, log names needed/available, no child survives, memory released, third start running on the stored plan, not remeasured); the lead's evidence page records it. The three-build contention repetitions and soaks were not run by decision (the change only alters the owner's reaction to the engine's own typed exit).

Evidence: docs/evidence/2026-11-26-pwr10-rsc14-native-acceptance (rerun on 2020.22.0 PASS).

## Change Log

- 2026-11-19T22:22:14.183742+00:00 (created-by): Created by agent
- 2026-11-19T22:22:32.846506+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:acceptance_criteria

## Reviews

- RV4661
- 2026-11-25T03:26:28.141785+00:00 (updated-by): Updated: section:notes
- 2026-11-25T03:33:56.082142+00:00 (updated-by): Updated: section:steps, section:acceptance_criteria
- 2026-11-26T05:43:22.437659+00:00 (updated-by): Updated: section:notes
- 2026-11-26T08:35:08.322269+00:00 (updated-by): Updated: section:notes
- 2026-11-26T08:35:32.130814+00:00 (state-transition): State: pending → completed
