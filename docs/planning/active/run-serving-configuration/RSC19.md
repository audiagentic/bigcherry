---
id: RSC19
order: 0
plan: run-serving-configuration
state: pending
created-at: '2026-11-27T12:56:15.096909+00:00'
breadth: ''
skill: ''
created-by: agent
work: M
priority: P1
---

# Component-owned configuration enhancements

## Description

Native YAML and upstream parser remain authoritative. Extend owned serving components and existing tuning/evidence owners; no new configuration layer or upstream patch for native settings.

## Steps

1. Seal managed environment; reconcile authored tuning projection. 2. Report mixed-step target, interleave mode/share, room/long-prefill/preemption controls, parsed defaults and resolved bring-up values. 3. Validate capability, pipeline, wire, KV and shard constraints. 4. Reuse exact kernel/provider/geometry/device/topology-qualified tuning; VTP60 and kernel composition own runtime implementation. Add a narrow engine seam only where native extension points are absent. 5. Audit duplicate variants with RSC12; qualify changes using existing quality and latency gates.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Matched native and owner builds; parser/receipt mutation tests, missing versus automatic settings, inherited environment, conflicting overrides, stale tune-cache identity, all-profile explain and representative GPU parity. No duplicate GPU campaign.

## Effort & Risk



## Standards



## Acceptance Criteria

One native YAML; no hidden overrides; authored, parsed and effective settings distinguished; all projected fields reconciled; no new patch for owned functionality; no unqualified production defaults.

## Notes

First slice: four native tuning options projected, existing projection reconciled, inherited RADIANCE_* isolated. Owner compiled; test execution blocked by ctest allowlist. Full native integration and remaining fields pending.

Owner build: bc_serving_test and bc_session_test linked. Matched Radiance host build: bc_native_fleet_config_test linked (222 build actions). Profile roster/variant/generator suite: 6 passed, 49 subtests passed. CTest execution denied by remote allowlist; C++ runtime/mutation and installed GPU gates remain pending. Changes confined to owned component and docs; no YAML production values or patch files changed. Reserved environment namespace now includes RADIANCE_*; use non-reserved names for API key references.

Second slice: native inspector v3 projects parse-time scheduler values and authorship (interleave steps/share, mixed-step target, room gate, long-prefill threshold, preemption wait). Distinguishes omitted, zero, auto and auto:1 without upstream patches. Strict owner validation and parser/mutation tests added. Effective bring-up receipt/manifest reconciliation and capability qualification remain pending; runtime test execution requires authorized runner.

CTest is authorized. Initial native parser and all four session groups passed; serving mutation gate exposed strict JSON rejecting floating-point shares. Fixed with round-trip decimal text, bounded numeric validation, strict-parser regression and valid auto-share admission tests; bc-serve explain now surfaces parsed scheduler. Rebuilt both owners: native test 1/1 passed, serving/session tests 5/5 passed (58.90s). Native /diagnostics already exposes configuration option/value/source and bring-up decisions; next slice should reuse it for effective-value reconciliation without a new engine patch. GPU/capability gates remain pending.

Coordinate with VTP59/VTP60: use existing MRC06 lab adapter capture_tuning_receipt against native /diagnostics, no second YAML parser. Reconcile direct manifest fields and additional explicit arm expectations, require matching pre/post receipts, retain through existing native lab ingestion. 26 targeted Python tests pass. Live exact 2.3 candidate (prior owner, new lab adapter) TP3 IQ8R FP8 KV 100/150ms targets: 4K/512 aggregate 179.990/186.543, max TTFT 15.782/14.604s, p95 gap 161.933/41.349ms; 512/128 aggregate 281.492/269.749, TTFT 2.139/2.166s, p95 gap 166.390/174.782ms. Four completed reports, zero preemptions, clean stops, identical prompt hashes/counts vs prior ABBA, both native receipts verified. Evidence docs/evidence/RSC19-VTP59-latency/summary.json. No production promotion; matched enhanced owner build, automatic expected settings from inspector, cross-field capability and numerical quality gates remain pending.

## Change Log

- 2026-11-27T12:56:15.096909+00:00 (created-by): Created by agent
- 2026-11-27T12:58:57.211030+00:00 (updated-by): Updated: section:notes
- 2026-11-27T13:14:50.590396+00:00 (updated-by): Updated: section:notes
- 2026-11-27T13:22:28.116046+00:00 (updated-by): Updated: section:notes
- 2026-11-27T13:39:37.473752+00:00 (updated-by): Updated: section:notes
