---
id: RHP18
order: 0
plan: patching-radiance-heat-placement
state: pending
created-at: '2026-11-10T03:10:34.325501+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# Live placement observability: the manifest, the launcher and the helper say whether the policy is on, why not, and what a calibration is waiting for

## Description

derive_manifest_live_policy returns nothing, silently, when the calibration store has no record for the configuration's calibration key. #790 added helper_budgets_mib to the plan key inputs on 2026-11-09, which orphaned the flashnext-4gpu record seeded on 2026-11-07: every production start of the layout since then ran with no live policy and nothing said so (found 2026-11-10 by checking the manifests: live_policy present until 2026-11-09, absent after; 35 plan-cache entries before #790 reproduce the stored key, none after). Placement A/Bs and kernel A/Bs measured in that window are policy-off.

## Steps

1. bc-serve writes spec.handoff.placement.live_policy_state = {state: on|off, reason} (no record for key K; record incomplete: why; calibration cache not given; layout not measured), so a manifest always says whether the policy is on and why not.
2. The launcher prints the same line at start and the benchmark/lab scripts refuse a placement-sensitive arm that expected the policy on when the manifest says off.
3. Decide whether the calibration key should keep covering helper_budgets_mib: a record measured for one tail budget is not valid for another (the reason #790 changed the plan key), so the key change may be right, but a layout change must then re-calibrate with a message, not silently disable.
4. Add a check (bigcherry check or a launcher preflight) that a layout marked calibrated has a record under today's key.
5. Re-calibrate flashnext-4gpu (recipe in docs/reference/engines/radiance/LIVE_PLACEMENT.md) and record the new record's evidence.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Unit test in serving_test.cpp for each off reason; manifest of a start without a record carries live_policy_state off with the key; a start with a record carries on; launcher output shows the line; a deliberate key-input change produces the message, not silence.

## Effort & Risk



## Standards



## Acceptance Criteria

No start can run with the policy off without the manifest and the launcher saying so and why; flashnext-4gpu has a record under the current key and a plain start carries live_policy.

## Notes

Operator guide: docs/reference/engines/radiance/LIVE_PLACEMENT.md. Re-calibration run: runs/recalibrate-4gpu (2026-11-10).

FINDINGS 2026-11-10 (why the policy was silently off or never calibrated): (1) the calibration key includes helper_budgets_mib since #790, orphaning the 2026-11-07 flashnext-4gpu record; no message. (2) Any start with --experiment (lane trace, profile_ops, kv_resident...) gets the lab service profile, and derive_manifest_live_policy runs only for the production profile, so an instrumented run never carries the policy unless --live-policy is given explicitly; no message. (3) The experiment hash is also part of the calibration key. (4) A missing record is silent; only an incomplete record logs a skip reason. (5) The helper prints its calibration waiting list only when it changes, and the calibration loop has no bound: a 2026-11-10 run served 1,600 requests over 80 minutes with 8 committed moves and never emitted its record. Interim: the Nov-7 record was migrated to the new key (store backup in runs/recalibrate-4gpu). Operator guide: docs/reference/engines/radiance/LIVE_PLACEMENT.md.

GATEWAY REVIEW 2026-11-10 (verified findings): (a) default heat period: the launcher passed handoff.histogram_period_s=60 while the derived policy needs heat younger than ~10 s (cadence ~5 s): a default start carried the policy with a stale feed (26 no_live_feed lines, 6 evaluations, 0 moves in ~15 min; runs/histogram-period-default); FIXED in #802 (launcher default 1 s; the period is not in the plan or calibration key). Still to do: bc-serve refuses or derives a period inconsistent with the policy (max_heat_age/cadence) instead of trusting the launcher. (b) Manifest contract v1 is closed (serving/contract/README.md): live_policy_state needs a contract-version decision; stage 1 records the startup decision in the existing /spec/handoff/placement provenance entry (requested on/off/not_applicable, configured, reason_code, reason_detail, calibration_key, record identity, effect_gate, histogram period, cadence, max_heat_age) plus a launcher line; runtime per-device state is a separate snapshot keyed by manifest digest + epoch + owner-generation digest. (c) Calibration key v2: explicit canonical measurement-relevant inputs (model/format/stride, helper count/order/arch/role/formats/actual budgets, engine topology and TP ranks, initial placement strategy and heat ranking identity, runtime geometry, kernel/transport compatibility, heat-source semantics and cadence, objective/workload class); trace-only experiments must not change it; no automatic fallback to a historical record; an explicit migration utility reuses a record only after reconstructing and comparing every measurement-relevant field (the 2026-11-10 manual migration of the flashnext-4gpu record is of that kind). (d) Preflight shared by bigcherry check, the launcher and benchmarks: layout/model/digest/measured plan/calibration key; policy expectation on|off|not_applicable (explicit, recorded); calibration match; topology compatibility; histogram period/freshness vs derived policy; owner-generation agreement; benchmark results labelled policy-on/off/not-applicable; policy-off arms use an explicit recorded option, never a moved store. (e) Runtime can stop migrating after a cleanup fault (worker_epoch.h:441-457) while the start still reads configured-on: report degraded/placement_stopped separately.

## Change Log

- 2026-11-10T03:10:34.325501+00:00 (created-by): Created by agent
- 2026-11-10T04:35:32.723532+00:00 (updated-by): Updated: section:title, section:notes
- 2026-11-10T05:24:00.607447+00:00 (updated-by): Updated: section:notes
