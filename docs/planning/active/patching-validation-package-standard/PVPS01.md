---
id: PVPS01
order: 0
plan: patching-validation-package-standard
state: pending
created-at: '2026-09-09T11:01:25.029179+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P1
---

# PQM-v1: standard Patch Qualification Matrix (4 arms x N architectures, isolated + in-situ)

## Description

Complete PQM-v1 as the standard patch qualification matrix: four contrast arms across gain/safety architectures, isolated and in-situ, with fail-closed planning, execution, evidence, and admission gates.

## Steps

1. Generate a versioned matrix from existing contract and architecture declarations: llama-native, BigCherry-native, BigCherry-native+P, and full release R/R+P; derive gain_arches from contract scope and safety_arches from validation coverage. 2. Resolve release-delta subject as exact current release plus P's dependency closure; fail closed on unknown/unavailable architecture, contract mismatch, or non-composable shipped patch. 3. Run gain architectures with 3v2 superiority and 4v4-P non-inferiority; safety-only architectures get correctness, activation, attestation, one smoke each, and a preregistered escalation trigger. 4. Cache binaries/build artifacts but never reuse historical measurements as paired evidence; rerun controls inside every paired session and rerun 2v1/4v1 only at release/pin/framework changes. 5. Execute all cells even after a failure for complete diagnostics; merge correctness so any failing result wins and any cell error yields invalid fail-closed verdict. 6. Require VA25 execution attestation and a complete generated matrix before release admission; defer multi-GPU/topology until all single-GPU gates and identity/visibility/topology prerequisites pass.

## Detailed Solution & Technical Design

PQM-v1 is a global policy identity bound by version/hash. The planner emits patch_id, contract_id/hash, gain/safety architectures, isolated and release compositions, release_composition_hash, and policy. The generic composition resolver uses exact overlays and dependency closure, never patch-specific campaign engines. The execution orchestrator runs every cell serially, separates inferential effects from safety smoke cells, merges correctness across cells with failures taking precedence, and produces invalid on infrastructure errors. The gate distinguishes isolated superiority from in-situ non-inferiority. Multi-GPU phases are explicitly later and require device/topology/P2P/collective identity.

## Code Samples & Guidance



## Files

campaign/resolution.py; campaign/qualification_matrix.py; campaign/qualification_execution.py; campaign/qualification_rd08.py; patch validation/evidence schema and admission gates; tools/tests/campaign/test_qualification_matrix.py, test_qualification_execution.py, test_qualification_rd08.py; VA25 attestation integration.

## Validation

Run offline matrix/execution/adapter tests and a real RD08 gfx1100 parity adapter run using an isolated clone, with control/subject compositions and attestation. Verify the known bit-identical correctness failure and near-zero combined effect reproduce the existing hand-rolled verdict, proving orchestration parity rather than falsely passing. Verify fail-closed planner paths, stale release composition detection, non-composable counterfactual handling, smoke exclusion from promotion, complete diagnostics after a cell failure, and no historical measurement reuse. Do not claim full PQM or multi-GPU completion until live adapters and all declared architectures are run.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

PQM-v1 matrix generation is deterministic, version-bound, fail-closed, and derived from existing declarations. Four arms and isolated/in-situ contrasts are represented with correct superiority/non-inferiority semantics. Execution is attested, runs all cells, merges failures safely, and rejects infra errors; evidence verifier requires complete matrix. Historical measurements never substitute for paired observations. Multi-GPU remains gated behind single-GPU correctness/attestation and topology identity.

## Notes

Supersedes: VA26
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-validation-package-standard-va26

Supersedes VA26. Preserve VA25 dependency, RD08 parity evidence, release composition hashes, admission ordering, and explicit deferred work (multi-GPU, calibration scheduling, cache layer, automatic escalation, reverse-dependency index, and schema enforcement) until separately implemented.

## Change Log

- 2026-09-09T11:01:25.029179+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:18:22.560586+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.611228+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:47.484054+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:48:13.282693+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria, section:notes
- chg_20260910_034835_repaired-the-final-six-live-su_3009
- 2026-09-10T03:48:35.219417+00:00 (updated-by): Updated: section:ledger-events
