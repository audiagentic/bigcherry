---
id: MRC04
order: 4
plan: run-measurement-records
state: completed
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

Merged: #1169 (narrow policy calibration key shared by calibration run and production), #1199 (platform calibration-record v1 schema, declared kinds and keys, reader/validator, bigcherry calibration list|show|validate, native Radiance adapter), #1220 (typed calibration reference/quantity blobs in run records; native serving tests: a record written on one build activates on another for the same narrow key), #1223 (strict readers: hash-only rows refused; mandatory provenance on native ingest; bc-serve calibration-prune applied to the live store, 6 dead rows removed and kept verbatim as a test fixture). Tests: tools/tests/core/test_calibration_records.py, bc_live_policy_test, bc_live_policy_ingest_test, bc_serving_test.

## Effort & Risk

Medium. Depends on the Radiance key fix landing first; risk of two parallel key definitions, mitigated by building on its result.

## Standards



## Acceptance Criteria

Calibration records are keyed by a declared narrow key and carry full provenance; a record survives rebuilds and activates on any build with the same key; readers refuse unknown or hash-only formats instead of keeping a legacy path; the production store holds only current records.

## Notes

First user landed (#1169, 2026-11-19): the Radiance live policy's calibration record is keyed by the narrow policy-calibration key v2 (`policy_calibration_inputs`: model, layout, objective, power profile, engine and helper cards, histogram period, helper runtime, remote service) and stores `calibration_inputs`, `source_plan_key` and `source_engine_build` as provenance; engine build, host pool, helper budgets and flags are excluded, so records carry across rebuilds and bumps; legacy hash-only records are refused. This generalises into the platform store: keep that declared-key shape, move the TOML record behind the platform reader, and make 'which record did this run use' a field of the run record. Tests to reuse: the calibration-run key equals the production lookup key, including a re-measured host pool (`serving_test.cpp`).

## Change Log

- 2026-11-19T00:50:37.155384+00:00 (created-by): Created by agent
- 2026-11-19T00:54:17.503543+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:effort_risk, section:acceptance_criteria
- 2026-11-19T07:39:03.286068+00:00 (updated-by): Updated: section:notes

## Slice 1 implementation note (2026-11-20)

The platform defines calibration-record v1 and declared key input names under
`tools/bigcherry/core/schemas/`. Native Radiance TOML rows remain the
production format, with a versioned adapter and no change to key v2 from #1169.
Platform JSON store write operations are lock-serialised, atomic and durable,
and genuinely hash-only legacy TOML rows are inspectable but never
activatable. Post-#1169 scoped rows with `calibration_inputs` and source
plan/build identities remain current and activatable via an in-memory adapter
even without the new MRC04 fields; the TOML store is never rewritten on read.
Platform `bigcherry calibration list|show|validate` is read-only.
The remaining MRC04 slices are MRC01 quantity-blob interop for large evidence,
run-record `calibration_record` reference migration, and device activation
proof across separate builds; these are not claimed by the CPU-only slice 1.
- 2026-11-19T22:35:13.458602+00:00 (updated-by): Updated: section:validation, section:acceptance_criteria
- 2026-11-19T22:35:17.919941+00:00 (state-transition): State: in_progress → completed
