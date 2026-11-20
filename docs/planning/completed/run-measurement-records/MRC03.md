---
id: MRC03
order: 3
plan: run-measurement-records
state: completed
created-at: '2026-11-19T00:50:32.612787+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Engine feature-state contract: every engine reports optional features active/inactive with a reason; platform surfaces it in status, records and production warnings

## Description

A feature-state contract: each engine declares its optional performance features and reports each as active, inactive or degraded with a reason code. The platform shows the states in status, copies them into every record, and warns loudly at production start when a production feature is inactive. Motivation: the Radiance live placement policy was silently off from 2026-11-09 to 2026-11-19 (the calibration key was orphaned by #790), costing about 32 % of cold 27K TTFT; nothing flagged it until a transaction count was compared by hand.

MRC03 scope: Radiance's live placement policy, MTP, prefix cache, KV host tier, helper combine and vision options. The llama.cpp feature-state producer and registry coverage are deferred to MRC07; no llama.cpp adapter is introduced here.

## Steps

- 1. Contract: a schema for `feature_states` (feature id, state active|inactive|degraded, reason_code, detail) and a registry of the features each engine declares, owned by the platform.
- 2. Radiance adapter: bc-serve writes the states file into the run directory at start (the native runtime stays Python-free). Map the existing manifest `spec.handoff.placement.live_policy_state` (configured, requested, reason_code, reason_detail) and add MTP, prefix cache, KV host tier and helper combine from the resolved configuration.
- 3. Deferred to MRC07: llama.cpp adapter and its engine-specific registry entries. MRC03's implementation and validation cover Radiance only.
- 4. Platform: show states in `bigcherry status`, copy them into every record (MRC01), and at production start print a loud warning (and a non-zero exit under a strict flag) when a feature marked production is inactive.
- 5. Tests: contract validation, adapter fixtures, warning on an inactive production feature.

## Detailed Solution & Technical Design

The engine runtime does not link Python: bc-serve (C++) writes the schema file; the platform reads it. The live policy is the proving case: serving already computes live_policy_state in manifest_writer.cpp, prints reason_code/reason_detail at startup in main.cpp and checks it in runtime_guards.cpp, so the Radiance adapter is largely serialising state that exists; the new part is the platform consumer and the warning.

## Code Samples & Guidance



## Files

- engines/radiance/components/serving/manifest_writer.cpp (writes spec.handoff.placement.live_policy_state)
- engines/radiance/components/serving/main.cpp (prints live_policy_state reason code and detail at start)
- engines/radiance/components/serving/runtime_guards.cpp (checks handoff.placement.live_policy_state)
- tools/bigcherry/core/run_manifest.py (run directory the states file lands in)
- docs/evidence/2026-11-19-flashnext-4gpu-regression/README.md (cause 1)

## Validation

Merged: #1195 (bc-serve writes native feature states at startup; production WARNING when the live policy is not active), #1230 (platform feature registry declaring Radiance feature ids and production expectations; native snapshots feed status and measurement; incomplete or wrong snapshots refuse publication; bc-serve --strict-production-features). Hardware: no---set production smokes of installed bc-radiance-1170.33.0 and 1170.34.0 wrote feature-states.v1.json with 5 active features (live_policy configured) and no warning. Tests: tools/tests/core/test_feature_registry.py, test_feature_states.py, serving feature_state_test and session tests; core/bench/lab suites pass.

## Effort & Risk

Medium. Risk: a warning noisy enough to be ignored; only features marked production warn.

## Standards



## Acceptance Criteria

Radiance's optional performance features are declared in a platform registry with production expectations; every Radiance start reports each feature's state and reason natively; records carry complete feature states; a production start with an expected feature inactive warns (or refuses under strict), so a silently-off live policy cannot recur. llama.cpp feature states are tracked in LCM02.

## Notes

First increments on main (2026-11-19): the heat guard and runtime receipt refusals now name the failing check instead of an enum number (#1174), the manifest's `live_policy_state` reason_code/reason_detail is printed at start, and a start without a usable calibration record prints `WARNING live policy OFF: policy-calibration key <key>; <reason_code>: <detail>` (#1169). These are the Radiance adapter's inputs for the platform feature-state file; the platform consumer (status, records, production-start warning) is still to build. RHP18 (the native-side observability) closed on this basis.

## Change Log

- 2026-11-19T00:50:32.612787+00:00 (created-by): Created by agent
- 2026-11-19T00:54:13.072249+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:effort_risk, section:acceptance_criteria
- 2026-11-19T07:38:59.061941+00:00 (updated-by): Updated: section:notes
- 2026-11-20T01:54:00.086087+00:00 (updated-by): Updated: section:validation, section:acceptance_criteria
- 2026-11-20T01:54:04.546515+00:00 (state-transition): State: pending → completed
