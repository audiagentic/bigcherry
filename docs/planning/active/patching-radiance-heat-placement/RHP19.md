---
id: RHP19
order: 0
plan: patching-radiance-heat-placement
state: in_progress
created-at: '2026-11-10T04:35:36.821205+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Calibration that always terminates with a usable record: bounded loop, periodic waiting report, and a deliberate tail-lane move to measure the tail-lane quantities

## Current evidence reconciliation (2026-11-10)

PR #801 repaired report newlines and the original 30-minute run had a COMPLETE record, not a stuck tail watchdog. The finite/partial lifecycle and passive event measurement gates remain open.

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

## Card-free execution status and lease-only acceptance (2026-11-10)

Merged #801 already fixed the newline issue; the original run emitted a COMPLETE marker at about 30 minutes, so calling it a tail-watchdog hang is wrong. The following status is per step, not an assertion of full calibration success.

1. Offline no-tail-transaction reproduction: **open**. Synthetic marker/partial-record card-free tests now verify non-success and absent-reason behavior, but a genuine replay trace with no proposed tail move is still needed.
2. Deliberate compatible forced tail-lane move through the RHP11 epoch harness, safely restored and marked calibration probe: **open**, requires offline state-machine proof before any hardware activation.
3. Waiting reports: **launcher card-free slice done**. During calibration it periodically prints the latest helper waiting reasons after completed client requests, even if the helper's list is unchanged; helpers still need per-condition elapsed/timer-armed detail. Reporting is bounded by the HTTP client's existing per-request timeout.
4. Finite lifecycle: **launcher card-free slice done**, not the entire helper lifecycle. The outer envelope is four A/B/B/A phases times derived minimum-sample warmup, heat-age and residence horizons, with no arbitrary literal timeout. It fails with the last reported missing reason. INITIALISING/COLLECTING/MEASURING_EVENTS/SETTLED/PARTIAL/FAILED helper states and first-feed-independent per-condition deadlines remain open.
5. Complete marker: **launcher card-free slice done**. Exactly one parseable record with every planner-required quantity is needed. Effect-only absent quantities remain allowed; the existing strict bc-serve calibration-ingest validates current manifest, key, objective, layout and atomically writes the store. A partial record cannot be installed by this launcher path.
6. Per-layout calibration matrix: **hardware pending**, see RHP20. No claim of calibrated policy on other card arrangements.

Card-free tests: tools/tests/lab/test_rhp19_calibration_watchdog.py and tools/tests/lab/test_rhp16_calibrate.py. Retain the helper's measured-drift watchdog as an independent second line; the launcher envelope is a fail-closed policy-derived bound rather than a proof that every passive event will become measurable. Its successful limit is not yet established across the layout matrix.

### Lease-only hardware acceptance commands

Set BUILD to the matching engine/component build, POLICY_JSON to the complete explicit measured lab policy, PROMPTS_A and PROMPTS_B to the distinct recorded long filler and document prompt lists. Start from repository root within the GPU lease; preserve the manifest, log, calibration key, and result for each arm. Stop and check all processes exited before the next layout.

| Layout | Command | Gate | Expected record |
| --- | --- | --- | --- |
| flashnext-4gpu normal convergence | bash tools/lab/radiance/launch-flashnext-best.sh start --build "$BUILD" --layout flashnext-4gpu --record-period 1 --calibrate --live-policy "$POLICY_JSON" --calibration-prompts-a "$PROMPTS_A" --calibration-prompts-b "$PROMPTS_B" | Self-terminates inside printed policy-derived bound with exactly one COMPLETE planner record; manifest/key verified by calibration-ingest; subsequent plain start ON | Work-root live-policy-calibration.toml, manifest JSON, helper0 log, actual key |
| flashnext-2xtx-rdna2-tail | bash tools/lab/radiance/launch-flashnext-best.sh start --build "$BUILD" --layout flashnext-2xtx-rdna2-tail --record-period 1 --calibrate --live-policy "$POLICY_JSON" --calibration-prompts-a "$PROMPTS_A" --calibration-prompts-b "$PROMPTS_B" | Tail-involved committed transaction or deliberate compatible tail probe; no missing tail-lane planner fields; all source/target formats valid | Separate calibration key/record, helper log, epoch/transaction receipt |
| flashnext-4gpu feed absent (fault injection, lease-held dedicated run) | Re-run the first command with a test harness that suppresses histogram publication, **never on the production service** | FAILED finite bound naming heat_watermarked, no calibration-store mutation | Failure stderr, helper waiting report, unchanged store digest |
| flashnext-4gpu no natural tail move (fault-injected harness) | Replay the same workload via RHP11 forced-move harness only after a deterministic restore test exists | Probe recorded distinctly from policy choice; planner record complete or explicit PARTIAL failure, no owner-map divergence | Forced epoch receipt, original/final owner-generation digests, helper log |

The first two commands are supported launcher entry points. The two fault-injection/harness rows are acceptance requirements, not executable production CLI flags: no safe completed forced-probe command exists yet, and inventing one would bypass RHP11's restoration gate.

## Change Log

- 2026-11-10T04:35:36.821205+00:00 (created-by): Created by agent
- 2026-11-10T05:23:56.608044+00:00 (updated-by): Updated: section:notes

## 2026-11-11 hardware finding: measured drift horizon omitted from launcher envelope

runs/rhp18-acceptance/calib.log: flashnext-4gpu, build 674f08546ca07ab7f9e9be15d1193217, 1s route feed, explicit policy/prompt A/B; launcher reported a 240s "policy-derived" deadline and FAILED at 240s with only drift_window and drift_half_life_observations missing. The helper0-1111-093708.log subsequently contains a COMPLETE marker; prior runs/recalibrate-4gpu had emitted one at about +30min (previous marker-framing bug). A 240s deadline is not evidence of helper failure.

Cause: calibration_watchdog.py covered minimum samples, heat age and residency, but omitted the observation-spaced retained DriftTracker horizon. worker_policy_calibration.h derives its own watchdog basis from observed accepted heat gap * drift capacity; the ring coarsens when full. Fix: the helper now reports structured measured drift capacity, gap, observation rate and retained horizon; after the first measured heat gap the launcher extends and freezes its finite A/B/B/A bound using the larger of capacity * max(gap,cadence,heat age) and observed horizon/rate. No guessed wall time. Without measured feed it retains the policy-only finite deadline. Even with the extension, absent half-overlap crossing must end with named missing conditions rather than an endlessly sliding bound.

Acceptance still hardware-pending: rebuild engine/helper from this slice, re-run same flashnext-4gpu calibration within lease, save progress/deadline/COMPLETE marker, verify manifest+store key and subsequent ON + live epochs. Card-free unit and C++ syntax checks are the slice gate, not proof of hardware convergence.
