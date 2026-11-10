---
id: RHP19
order: 0
plan: patching-radiance-heat-placement
state: pending
created-at: '2026-11-10T04:35:36.821205+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Calibration that always terminates with a usable record: bounded loop, periodic waiting report, and a deliberate tail-lane move to measure the tail-lane quantities

## Description

launch-flashnext-best.sh --calibrate loops until the helper prints RHP16_CALIBRATION_JSON. The helper's timeout machinery (worker_policy_calibration.h CalibrationProgressTimeout) only starts a quantity's timer once it is measurable; tail_lane_service_ns_per_s, tail_lane_row_ns and lane_lost_ns are measurable only after a committed transaction that involves the tail lane (m.tail_committed_transactions > 0). If the policy never proposes such a move (835 of 845 evaluations on 2026-11-10 found no improvement above the margin because the startup placement already fits both prompt sets), no timer starts and the loop never ends. It also prints the waiting list only when it changes, so the operator cannot see what is missing. The 2026-11-07 calibration ended with a partial record after the planner timed out.

## Steps

1. Reproduce offline: r3_policy_replay over a recorded heat series with a placement that already fits both workloads: confirm no tail-involved transaction and no timer.
2. The calibration run issues one deliberate cross-helper swap involving the tail lane through the RHP11 forced-move harness (slice G), once the feed is open, so tail_committed_transactions becomes positive; record that this move is a calibration probe, not policy output.
3. Periodic waiting report (every derived horizon fraction): quantity, reason, elapsed, whether its timer is running.
4. A bound derived from the helper's measured drift horizon: the loop ends with either a record or an error naming the missing quantities; never an unbounded wait.
5. Fix the completeness rule so a record missing only the effect quantities (allowed absent) is accepted exactly as on 2026-11-07, and the launcher validates it against this run's strict manifest as today.
6. Calibrate every layout/card combination in RHP20's matrix with it.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

worker_policy_calibration_test and replay cases: (a) tail move never proposed -> probe commits it and the record completes; (b) feed never opens -> bounded error naming heat_watermarked; (c) normal run unchanged. Hardware: flashnext-4gpu calibration ends by itself with a complete record under the current key and a plain start carries live_policy.

## Effort & Risk



## Standards



## Acceptance Criteria

Every --calibrate run ends with a complete record or an error that names the missing quantities, within a bound derived from measured quantities; flashnext-4gpu is calibrated under the current key by that path.

## Notes

Evidence: runs/recalibrate-4gpu (2026-11-10), helper0 log tail. Related RHP18 (observability), RHP20 (card-type matrix).

CORRECTION 2026-11-10 (gateway review, verified in the raw log): the 80-minute calibration run did not hang on the tail-lane watchdog. The helper emitted a COMPLETE record at about +30 min (8 committed moves, all planner and tail-lane quantities, only abort_quiescent_ns absent) but the 'calibration waiting' line was printed with a literal backslash-n (worker_policy_driver.h), so RHP16_CALIBRATION_JSON ended up mid-line and the launcher's anchored grep never matched. FIXED in #801 (calibration_report_line + framing test); re-run through the supported path: runs/recalibrate-4gpu-v2. The earlier hypothesis (no tail-involved move, so no timer) is a real liveness gap in CalibrationProgressTimeout (worker_policy_calibration.h:112-203: the transaction timer arms on the first commit, the tail-lane timer on the first tail-involved commit) but it was not what stalled this run. Remaining RHP19 work, in order: (1) launcher validates the marker (complete, once, bound to this manifest) and bounds the loop independently; (2) decouple passive quantities (tail-lane service ns/s, ns/routed row) from commits; the quiesce/reopen loss needs an epoch cycle (Controller::begin_epoch_cycle, no ownership change) and the copy bandwidth/transaction time need a real transaction; (3) finite lifecycle INITIALISING -> COLLECTING -> MEASURING_EVENTS -> SETTLED | PARTIAL | FAILED with per-quantity deadlines derived from measured cadence, partial records listing the missing quantities and never accepted as production-calibrated; (4) optional lab-only forced compatible swap probe through the production epoch path with proven restoration. Design question for a Fable-class session: gateway-drafted, held until the re-run shows whether the newline fix is enough.

## Change Log

- 2026-11-10T04:35:36.821205+00:00 (created-by): Created by agent
- 2026-11-10T05:23:56.608044+00:00 (updated-by): Updated: section:notes
