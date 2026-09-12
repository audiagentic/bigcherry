---
id: PEC02
order: 1
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

Complete the production half of the mandatory trigger-proof contract by using the repository's current Experiment Contract and campaign evidence owners. Populate TriggerEvidence from real telemetry and thread INVALID status through campaign/report validation; do not create another evidence authority or executor. This is an authority primitive for PA33 and later qualification work, with hardware evidence remaining a separate completion requirement.

## Steps

1. Use tools/bigcherry/experiment/contract.py as the canonical TriggerEvidence/evaluate_promotion_gate authority and tools/bigcherry/experiment/execution.py as the current marker-to-evidence producer; use tools/bigcherry/campaign/planner.py and tools/bigcherry/campaign/qualification_execution.py for the per-lane runtime path. Do not revive the stale root-level experiment_contract.py path.
2. Select and document one real telemetry source first (0700 coverage_counters is the preferred launch-count source), then map its on-disk fields to positive contract lanes.
3. Implement or complete the reader that constructs TriggerEvidence with candidate launches/route selections and rejects malformed or absent evidence.
4. Thread the reader through campaign planner and per-lane evaluation so evaluate_promotion_gate() receives trigger_proof for real runs.
5. Preserve existing semantics: positive lanes must trigger, control/boundary lanes are not required to trigger, empty evidence is INVALID, and INVALID short-circuits pass/fail evaluation.
6. Add fixture/negative tests, surface INVALID distinctly in report/release validation, and run one real contract artifact end-to-end when the hardware/telemetry environment is available. PA33 must consume this authority rather than inventing a schema.

## Detailed Solution & Technical Design

PEC02 closes the wiring gap left by the existing pure evaluation gate. Telemetry extraction must be schema-validated and provenance-bound to the same lane/signature; never infer trigger proof from benchmark completion or final logits. Keep backward compatibility only for explicitly legacy callers, while campaign-produced contracts require trigger evidence.

## Code Samples & Guidance



## Files

tools/bigcherry/experiment/contract.py; tools/bigcherry/experiment/execution.py; tools/bigcherry/campaign/planner.py; tools/bigcherry/campaign/qualification_execution.py; tools/bigcherry/patch/validation_campaign.py; contract/report/release validation; fixtures and tools/tests/campaign contract/evidence tests.

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

GPT roadmap provenance: request req_f7a013040828433c, same session ses_76206cac3e6b4be0, exact pushed bb20f104. PEC02 is an authority primitive before PA33. Reconcile these current owners and the existing TriggerEvidence implementation before coding; do not add another evidence schema, gate evaluator, runner, or contract engine. Hardware-free malformed/empty/mismatch tests can proceed before real telemetry artifacts; the real artifact remains required for closure.

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
- 2026-09-12T18:53:45.966189+00:00 (updated-by): Updated: order=1, section:description, section:steps, section:files, section:notes
- chg_20260912_185547_recorded-the-gpt-guided-non-vu_6004
- 2026-09-12T18:55:48.051370+00:00 (updated-by): Updated: section:ledger-events
- chg_20260912_191132_corrected-the-gpt-guided-imple_8803
- 2026-09-12T19:11:32.084083+00:00 (updated-by): Updated: section:ledger-events
