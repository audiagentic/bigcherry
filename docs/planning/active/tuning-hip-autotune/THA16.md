---
id: THA16
order: 0
plan: tuning-hip-autotune
state: pending
created-at: '2026-09-09T10:50:01.518345+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: P0
---

# Critical-path/slack vs contention hypothesis for isolated-vs-e2e tuning divergence

## Description

Determine whether isolated-versus-E2E tuning divergence is caused by critical-path/slack and contention effects, using the real GPU0/27B production path and controlled host/API evidence.

## Steps

1. Reproduce the GPU0/27B mismatch with matched binaries, workload, topology, and diagnostics settings.
2. Separate GPU kernel time from host dispatch/API and contention time using rotation-controlled rocprofv3/kernel/API traces.
3. Compare isolated and E2E arms with identical tuned/native provenance and cache state.
4. Quantify signature/launch shares, critical-path slack, overlap, and contention rather than inferring from family counts.
5. Test candidate mitigations only when the trace identifies an attributable mechanism; preserve native baseline and no-regression controls.
6. Record negative results and tool-fit limitations explicitly.

## Detailed Solution & Technical Design

Capability owner: tuning

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

HI172 trace tooling/evidence; E2E benchmark arms; rocprofv3/API analysis; contention and critical-path report

## Validation

Rotation-controlled dual-XTX evidence; kernel trace and HIP API trace; matched native/tuned controls; host dispatch overhead attribution; predicted-vs-measured E2E reconciliation; clean shutdown and provenance receipts.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

The divergence mechanism is either evidenced or explicitly unresolved with bounded negative results. No tuning winner is promoted from isolated microbenchmarks when the critical-path/E2E contribution is unproven.

## Notes

Supersedes: HI172
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-hip-autotune-hi172

Supersedes: HI172
Inherited semantic scope: preserve GPU0/27B mismatch, kernel-versus-host attribution, rotation-controlled traces, and no-promotion-without-critical-path evidence.
Migration: capability-rebaseline-v3-2026-09

## Change Log

- 2026-09-09T10:50:01.518345+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:06:13.456800+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.923695+00:00 (updated-by): Updated: section:ledger-events
- chg_20260909_143622_cleaned-up-the-tooling-registr_2452
- 2026-09-09T14:36:22.958572+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.491373+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:24:22.861057+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:acceptance_criteria, section:notes
- chg_20260910_022438_the-next-five-high-risk-tuning_6580
- 2026-09-10T02:24:38.752279+00:00 (updated-by): Updated: section:ledger-events
