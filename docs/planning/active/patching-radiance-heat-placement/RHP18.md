---
id: RHP18
order: 0
plan: patching-radiance-heat-placement
state: in_progress
created-at: '2026-11-10T03:10:34.325501+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# Live placement observability: the manifest, the launcher and the helper say whether the policy is on, why not, and what a calibration is waiting for

## Current evidence reconciliation (2026-11-10)

PRs #798/#802/#804 deliver guide, cadence and explicit no-policy reasons; #790 invalidated the former calibration record. This is partial: manifest/preflight/degraded/runtime status and every supported layout remain to be accepted.

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

## Card-free execution and lease-only acceptance (2026-11-10)

Reconciled: #798 operator guide; #802 one-second heat default; #804 v1 provenance reason; #805 unsupported arrangements refused; #801 calibration log framing. Merged commits are not layout acceptance.

- Step 1 PARTIAL: v1 stores the startup reason in /provenance, not in a typed live_policy_state. Typed on/off/not_applicable/degraded and runtime owner-generation keyed snapshots are for the v2 contract lane; do not change the closed v1 writer.
- Step 2 CARD-FREE DONE here: launcher fail-closed preflight rejects malformed/unknown state, policy-on heat cadence stale relative to derived cadence/max_heat_age, and policy expectations mismatched via BC_EXPECT_LIVE_POLICY=on|off|not_applicable. Helper-free prints NOT_APPLICABLE. Existing bench scripts need to declare expected state on each arm; not every bench is wired.
- Step 3 DECIDED/OPEN: budgets must remain in the calibration key; no fallback to historical records. Canonical key v2, workload semantics and trace-only experiment hashing remain open.
- Step 4 PARTIAL: launcher preflight checks startup state and heat freshness; bigcherry check, record identity, and runtime epoch-generation agreement remain.
- Step 5 HARDWARE: recalibrate flashnext-4gpu under today's exact key; verify normal plain start ON and live epochs committed, not only manifest presence.

### Lease-only acceptance runs

Set BUILD to the matching build directory, POLICY_JSON to the valid lab policy, PROMPTS_A and PROMPTS_B to the independent recorded workload lists. Run from repository root, **inside the GPU lease**; stop the launcher between rows.

| Layout and arm | Exact command | Gate | Expected record |
| --- | --- | --- | --- |
| flashnext-4gpu policy ON | BC_EXPECT_LIVE_POLICY=on bash tools/lab/radiance/launch-flashnext-best.sh start --build "$BUILD" --layout flashnext-4gpu --record-period 1 | Startup ON and policy cadence compatible; check helper owner generation and migrations throughout the run | Manifest under run/flashnext, startup stderr, helper0 log and calibration key |
| flashnext-4gpu stale heat | BC_EXPECT_LIVE_POLICY=on bash tools/lab/radiance/launch-flashnext-best.sh start --build "$BUILD" --layout flashnext-4gpu --record-period 60 | Must refuse before process launch, naming cadence/max age | Preflight stderr; zero serving PIDs |
| flashnext-2xtx unsupported | BC_EXPECT_LIVE_POLICY=off bash tools/lab/radiance/launch-flashnext-best.sh start --build "$BUILD" --layout flashnext-2xtx | Explicit OFF reason: last XTX is not a tail | Manifest provenance and startup line |
| flashnext-r9700 helper-free | BC_EXPECT_LIVE_POLICY=not_applicable bash tools/lab/radiance/launch-flashnext-best.sh start --build "$BUILD" --layout flashnext-r9700 | Explicit NOT_APPLICABLE, no helper migration | Manifest and startup line |
| flashnext-4gpu new calibration | bash tools/lab/radiance/launch-flashnext-best.sh start --build "$BUILD" --layout flashnext-4gpu --calibrate --record-period 1 --live-policy "$POLICY_JSON" --calibration-prompts-a "$PROMPTS_A" --calibration-prompts-b "$PROMPTS_B" | RHP19 finite termination, complete once-only record, subsequent plain start ON | live-policy-calibration.toml, exact key, helper log, measured manifest |

Hardware quality: pair policy-on and -off under RHP12 and RHP22; no startup ON state proves the policy has not degraded later.

## Hardware acceptance finding: drift window and key provenance (2026-11-11)

Read-only audit of `runs/rhp18-acceptance/calib.log`: a `flashnext-4gpu` launch on build `674f08546ca07ab7f9e9be15d1193217` reported **ON** with explicit lab policy, but the launcher terminated at 240 seconds with `drift_window` and `drift_half_life_observations` outstanding. This is **not** a successful calibration nor a policy-on plain-start acceptance. The helper log `${BC_WORK_ROOT}/flashnext/helper0-1111-093708.log` contains a complete marker near shutdown, but the launcher did not validate/ingest it: no store mutation or manifest-on claim may be inferred.

The key mismatch is **not an unexplained build-code change**. Comparing actual `plan-cache/*.json` key inputs for `0667fa1810a7...` (the record used with f17f21) and `7443c74886a1...` (the new build) isolates `engine.host_pool_mib` **9216 -> 2259**, from RSC03 derived host-pool sizing; model, arch, helper budgets [22528,22528,14336], heat digest, layout, experiment and engine settings otherwise match. `plan_cache.cpp::calibration_key` already excludes `engine_build` and `cache_version`, so an engine binary hash did not cause the split. **Keep the host-pool field in v1's key:** it is a measured engine resource/plan input, and cross-pool reuse is not proven equivalent. Consider omitting it in a future canonical v2 *only* after verifying measured plan and all placement/heat-relevant invariants; never silently migrate or fall back.

Card-free repair in this slice: launcher watchdog accounts for the real A/B/B/A workload and reads helper-reported drift capacity, heat gap and observation horizon, separately failing an absent feed. The missing-calibration/why-off path remains fail closed. **Hardware still pending**: rebuild the helper/launcher together, run the recorded calibration under the new key, verify strict ingest, then a plain start with manifest policy ON and live committed epochs. No GPU was used by this slice.

## Change Log

- 2026-11-10T03:10:34.325501+00:00 (created-by): Created by agent
- 2026-11-10T04:35:32.723532+00:00 (updated-by): Updated: section:title, section:notes
- 2026-11-10T05:24:00.607447+00:00 (updated-by): Updated: section:notes
