---
id: PRBE67
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:58:14.009489+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# HIP-GRAPH-002: Force graph recapture when FA stream topology changes

## Description

Force HIP graph recapture when FA stream topology changes, replacing selective bypass where stale topology is the proven root cause.

## Steps

Use PRBE66 instrumentation to identify stream/shape transitions; add recapture at the exact topology-change boundary; test repeated long-context failures and stable-topology controls; measure recapture frequency/cost and retained graph speedup; compare directly with PRBE66 bypass.

## Detailed Solution & Technical Design

Repair stale captured topology by invalidating/recapturing HIP graphs when stream mapping or FA shape topology changes. Preserve cached graphs for stable topology and avoid unconditional recapture.

## Code Samples & Guidance



## Files

HIP graph cache/update logic; stream-topology instrumentation; failure/replay and stable-control tests; recapture cost evidence.

## Validation

No failure and output parity; recapture count/cost, long-run stability, and retained graph speedup versus selective bypass.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Promote over PRBE66 only when repeated long-context runs are robust, output-identical, and recapture cost is bounded while stable graphs retain their speedup.

## Notes

Supersedes: RD84
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd84

## Change Log

- 2026-09-09T10:58:14.009489+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:15:20.253208+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.430745+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.261223+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:17:52.882598+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_031805_repaired-four-more-graph-and-v_2834
- 2026-09-10T03:18:05.362966+00:00 (updated-by): Updated: section:ledger-events
