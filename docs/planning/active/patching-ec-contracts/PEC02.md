---
id: PEC02
order: 0
plan: patching-ec-contracts
state: pending
created-at: '2026-09-09T10:47:36.919359+00:00'
breadth: ''
skill: ''
created-by: capability-rebaseline-v3
priority: null
---

# Mandatory trigger-proof requirement for performance contracts

## Description

Complete the production half of the mandatory trigger-proof contract: populate TriggerEvidence from real telemetry and thread INVALID status through campaign/report validation.

## Steps

- Select and document one real telemetry source first (0700 coverage_counters is the preferred launch-count source), then map its on-disk fields to positive contract lanes.
- Implement a reader that constructs TriggerEvidence with candidate launches/route selections and rejects malformed or absent evidence.
- Thread the reader through campaign_planner.expand_contract() and the per-lane evaluation path so evaluate_promotion_gate() receives trigger_proof for real runs.
- Preserve the existing semantics: positive lanes must trigger, control/boundary lanes are not required to trigger, empty evidence is INVALID, and INVALID short-circuits pass/fail evaluation.
- Run one real contract artifact end-to-end, add fixture/negative tests, and surface INVALID distinctly in report/release validation.

## Detailed Solution & Technical Design

PEC02 closes the wiring gap left by the existing pure evaluation gate. Telemetry extraction must be schema-validated and provenance-bound to the same lane/signature; never infer trigger proof from benchmark completion or final logits. Keep backward compatibility only for explicitly legacy callers, while campaign-produced contracts require trigger evidence.

## Code Samples & Guidance



## Files

tools/bigcherry/experiment_contract.py; telemetry reader for 0700/0810/0820/0830; campaign_planner per-lane pipeline; contract/report/release validation; fixtures and tests.

## Validation

Malformed/missing telemetry; zero positive launches; route-selected-only; controls/boundaries; real artifact end-to-end; INVALID report rendering; full offline contract suite.

## Effort & Risk



## Standards

Fail closed on untriggered candidates; distinguish INVALID from FAIL; no benchmark-completion inference; preserve backward compatibility only for legacy explicit callers.

## Acceptance Criteria

At least one real campaign contract consumes machine-derived TriggerEvidence; untriggered positive lanes are INVALID, not pass/fail; evidence is schema/provenance bound; tests cover malformed, empty, control and passing cases.

## Notes

Supersedes: EC18
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-ec-contracts-ec18

## Change Log

- 2026-09-09T10:47:36.919359+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:03:40.700843+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.753861+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.203609+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:36:27.738695+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_023638_the-two-ec-contract-successors_4025
- 2026-09-10T02:36:38.467423+00:00 (updated-by): Updated: section:ledger-events
