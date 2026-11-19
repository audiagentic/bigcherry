---
id: MRC03
order: 3
plan: run-measurement-records
state: pending
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

Examples: Radiance: live placement policy, MTP, prefix cache, KV host tier, helper combine. llama.cpp: dispatch cache, tuning.

## Steps

- 1. Contract: a schema for `feature_states` (feature id, state active|inactive|degraded, reason_code, detail) and a registry of the features each engine declares, owned by the platform.
- 2. Radiance adapter: bc-serve writes the states file into the run directory at start (the native runtime stays Python-free). Map the existing manifest `spec.handoff.placement.live_policy_state` (configured, requested, reason_code, reason_detail) and add MTP, prefix cache, KV host tier and helper combine from the resolved configuration.
- 3. llama.cpp adapter: report dispatch cache and tuning state from the build/run configuration.
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

- Contract tests; a fixture with the policy inactive produces the warning.
- On hardware: start Flash-Next with a deliberately orphaned calibration record; the warning names the policy and the reason code (reuse the failure of 2026-11-09 to 2026-11-19 as the test).

## Effort & Risk

Medium. Risk: a warning noisy enough to be ignored; only features marked production warn.

## Standards



## Acceptance Criteria

- An inactive production feature is reported at start-up and in `status`, without reading logs.
- Every record contains the feature states of the run.
- Adding a feature to an engine requires only a registry entry and an adapter line.

## Notes



## Change Log

- 2026-11-19T00:50:32.612787+00:00 (created-by): Created by agent
- 2026-11-19T00:54:13.072249+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:effort_risk, section:acceptance_criteria
