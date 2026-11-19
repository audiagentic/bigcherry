---
id: MRC04
order: 4
plan: run-measurement-records
state: pending
created-at: '2026-11-19T00:50:37.155384+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Generic calibration-record store: narrow keys plus provenance so records carry across rebuilds; Radiance live policy is the first user

## Description

A generic calibration-record store: records keyed by a narrow, declared key (only the inputs that change the calibrated quantities), with full provenance (build, plan key) stored inside the record, so a record carries across rebuilds and bumps. Owner rule: evidence carries across bumps; calibration is provenance, never a start gate. Radiance's live policy is the first user.

Today the Radiance store is $BC_WORK_ROOT/live-policy-calibration.toml (tested by live_policy_test.cpp). Its key is calibration_key() in engines/radiance/components/serving/plan_cache.cpp: the plan-key inputs with engine_build and cache_version removed. A separate GPT lane is making that key build-independent (brief live-policy-key-build-independent.md in the gpt-briefs folder); MRC04 generalises the shape that lane lands rather than duplicating it. Note the key still includes helper_budgets_mib, layout, engine args and tuning, so a budget change orphans a record; the declared-key rule is how that decision becomes explicit per calibration kind.

## Steps

- 1. Wait for / read the Radiance key fix and write the generic record shape from it: kind, declared key inputs (named list), key hash, quantities, provenance {build id, plan key, source revision, date, run id}.
- 2. Platform store: put/get/list by (kind, key), provenance never part of the key, atomic writes, stored through MRC01 blobs where the quantity file is large.
- 3. Declared-key check: a kind declares its key inputs; a test refuses a key that includes build identity.
- 4. Radiance adapter: live policy reads and writes through the store (the C++ side keeps its TOML reader; the platform tool ingests and lists the same file until the native side is switched).
- 5. Surface in status: which calibration record a run used, or `none` with the reason.

## Detailed Solution & Technical Design

The store is an index over records, not a gate: lookup misses are reported and the engine proceeds. A calibration record found for a different build is used and flagged by provenance (build differs), not rejected. What is in the key is a per-kind decision recorded in code and in the kind's declaration.

## Code Samples & Guidance



## Files

- engines/radiance/components/serving/plan_cache.cpp (plan_key_inputs(); calibration_key(): erases engine_build and cache_version, keeps helper_budgets_mib)
- engines/radiance/components/serving/live_policy_test.cpp (current calibration file test)
- docs/planning/active/patching-radiance-heat-placement/RHP16.md (calibration procedure: bc-serve calibration-ingest --calibration live-policy-calibration...)
- <dev-root>/tmp/bigcherry/gpt-briefs/live-policy-key-build-independent.md (the key-fix brief; scratch, not a repo file)

## Validation

- Unit tests: same key across two build ids returns one record; a key input that is not declared changes nothing; provenance preserved.
- Live: calibrate on one Radiance build, start a second build with the same layout, and observe the record is used and reported as carried across builds.

## Effort & Risk

Medium. Depends on the Radiance key fix landing first; risk of two parallel key definitions, mitigated by building on its result.

## Standards



## Acceptance Criteria

- A rebuild or pin bump never orphans a calibration record.
- Each record carries build, plan key and source revision as provenance.
- The declared key of each kind is listed in the docs and enforced by a test.

## Notes



## Change Log

- 2026-11-19T00:50:37.155384+00:00 (created-by): Created by agent
- 2026-11-19T00:54:17.503543+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:effort_risk, section:acceptance_criteria
