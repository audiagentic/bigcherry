---
id: VTP60
order: 0
plan: run-dense-27b-virtual-tp
state: pending
created-at: '2026-11-27T10:22:17.436914+00:00'
breadth: ''
skill: ''
created-by: agent
priority: P1
work: M
---

# Promote fast scheduler and model-layout qualification to canonical gates

## Description

Turn VTP59 host-burst proof into repeatable, provenance-bound fast tests. Reuse actual native scheduler, CTest, RCS26–28 affected-family resolution, existing replay/cost captures, MRC08 and MEN17. Do not own production tuning or replace integrated serving gates.

## Steps

S1 consolidate native host build/CTest: retain regression assertions and retire ad hoc link launcher; validate fresh combined run. S2 native effective configuration and workload descriptors, no duplicated defaults. S3 reuse rad-tune for declared kernel variants, rad-kbench for independent correctness and isolated replay, rad-info for model/graph scope and native --diagnose for actual phase/serving costs. Extend upstream sched_test sim_burst minimally for mixed-step timing, injected clock, versioned costs, structured schedule records; reuse actual scheduler and existing VTP42 governor. Deterministic policy search is orchestration only; RCS26–28 select affected families. S4 route receipts through MRC08 and MEN17 exact candidate gates. S5 mutation, repeated schedule hashes, held-out hardware correlation, final quality; retire redundant adapters only after equivalent coverage and evidence are verified.

## Detailed Solution & Technical Design

See docs/design/radiance-fast-qualification.md sections 1–9: exact native CTest, effective defaults, workload selection, measured-cost replay, evidence routing and acceptance. S1/S2 independent of GPU and MRC08; S3–5 consume existing owners. Review findings incorporated.

## Code Samples & Guidance



## Files

docs/design/radiance-fast-qualification.md; existing owners only. VTP59 retains tuning, RCS26–28 retain kernels, MRC08 retains record/export and MEN17 retains final acceptance.

## Validation

Native exact build, fake linkage, no model/GPU; fail missing/skipped tests, stale core, malformed cost and unknown geometry. Mutation controls for known overlap loss, default drift, slot/admission/cancel/count; validate cost uncertainty, actual row evidence and held-out hardware. Verify existing record/export privacy, exact candidate/reference, failed host gate prevents GPU. See design §8.

## Effort & Risk



## Standards



## Acceptance Criteria

- [ ] Exact composed native CTest build, effective default gate, missing/skipped fail, mutation controls and fresh combined run.
- [ ] Use native rad-tune for declared kernel variants and rad-kbench for independent correctness/replay; retain RCS family selection and demonstrate equivalent coverage before removing wrappers.
- [ ] Extend native simulator rather than fork it; identical frozen scenario, cost table, clock, seed and policy produce identical schedule hash and metrics across repeated runs; missing/uncertain costs nonpass for qualification.
- [ ] Native diagnose calibration and held-out hardware correlation by model/layout, with predeclared throughput/latency/fairness budgets and independent numerical quality.
- [ ] Existing MRC08 and MEN17 exact candidate/reference, privacy, coverage and fail-closed gates; no duplicate tuner, controller, workload catalog or evidence store.

## Notes

Owner direction: reuse Radiance tooling first. Native 2.3 source slice 90bca121c9dcee4cac6cf2b41a8d7b3e verified; commit 724ebe40c replaced manual linking with attested native CMake/CTest. Seven native cases, 18,221 assertions, 0.21 s rerun; 31 Python checks; injected default drift rejected and restored. Initial host build 151 s. Fresh combined launcher and full profile audit remain. Upstream sim_burst supports seeded acceptance and configurable linear costs; mixed target and clock not yet wired. Prefer upstream-compatible changes over new BigCherry implementations. Evidence docs/evidence/VTP60-native-host/summary.json.

## Change Log

- 2026-11-27T10:22:17.436914+00:00 (created-by): Created by agent

## Reviews

- 2026-11-27T10:25:47.614063+00:00 (updated-by): Updated: section:detailed_solution, section:validation, section:acceptance_criteria
- RV4753
- 2026-11-27T10:53:09.767000+00:00 (updated-by): Updated: section:steps
- 2026-11-27T10:53:26.112245+00:00 (updated-by): Updated: section:notes
- 2026-11-27T10:53:34.034298+00:00 (updated-by): Updated: section:acceptance_criteria
