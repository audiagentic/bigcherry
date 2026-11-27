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

## Change Log

- 2026-11-27T12:56:15.096909+00:00 (created-by): Created by agent
- 2026-11-27T12:58:57.211030+00:00 (updated-by): Updated: section:notes
