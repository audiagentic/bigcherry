---
id: RRBC03
order: 2
plan: run-reusable-build-campaign
state: completed
created-at: '2026-09-09T10:59:22.190994+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: P1
---

# Environment-action campaign inputs for non-patch experiments (driver/ICD/ASPM/hipBLASLt)

## Description

The frozen reusable-build-campaign item remains active with implementation/acceptance work outstanding and no terminal disposition.

## Steps

1. Implement the still-valid future scope.
2. Run the stated acceptance and evidence gates.
3. Preserve predecessor provenance and record successor evidence under this ID.

## Detailed Solution & Technical Design

Capability owner: run

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

tools/bigcherry/environment_inputs.py (new); tools/tests/test_environment_inputs.py (new). Reuses tools/bigcherry/tuning/journal.py (canonical/checksum), tools/bigcherry/experiment/bundle.py (SECRET/file_hash/safe_environment). NOT yet wired into: tools/bigcherry/campaign/build.py, campaign/lane.py, core/artifacts.py, core/provenance.py (RE37's original integration points) -- typed model landed, campaign-plumbing integration remains open.

## Validation

DONE 2026-09-10: 11 unit tests in tools/tests/test_environment_inputs.py, all passing -- determinism, secret-value rejection, artifact logical-role-not-path keying, scope partitioning (build/run/both/evidence_only), duplicate-name rejection, reversible-action restored-outcome enforcement, order-independent canonical document. Full offline suite (tools/tests/) confirmed green after this change. Remaining: no real hipBLASLt/rocblas-gemm-tune file has been captured through this model yet (HI173 integration is the real acceptance test, not yet run).

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Close the recorded gap and pass the frozen scope's stated implementation, correctness, performance, or evidence gate.

## Notes

Supersedes: RE37
Migration: capability-rebaseline-v3-2026-09
Successor key: run-reusable-build-campaign-re37

External dev-gpt holistic review (2026-09-10, req_9f60aaa2ae5f4b88): MINOR DESIGN REVISION then GO. EnvironmentInput concept is sound but one digest cannot blindly enter both build and run identity — add explicit projection/scope field: build | run | both | evidence_only. Offline tuning files (e.g. hipBLASLt/rocblas-gemm-tune solution caches) should be keyed by logical-role + content/artifact digest, not host-path identity. Mutating actions require requested-action plus pre/post observation and restoration outcome. This item is confirmed as the durable home for HI173's tuning-file evidence. Relationship to RRVP03 clarified: deliberately SEPARATE, not merged — RRBC03/EnvironmentInput = known/declared BEFORE execution (input to reproducibility/run identity); RRVP03/RuntimeStackAttestation = observed AFTER process launch + provider warmup (evidence validating what actually loaded). Do not put RRVP03's fingerprint into EnvironmentInput — a run evidence record should carry both as separate fields (environment_input_digest, expected_resolved_stack_fingerprint, build_stack_fingerprint, actual_runtime_stack_attestation_fingerprint) sharing canonical serialization/hashing primitives but remaining distinct schemas. Execution order: ranked #2 (directly unblocks durable HI173 evidence).

CORRECTION from deeper repo-validated dev-gpt review (2026-09-10): core kind/scope split is confirmed, but RETRACT the prior review's "one canonical hashing primitive across RRBC03/RRVP02/BRVP01/RRVP03" recommendation -- this project's existing build and experiment identities intentionally use separate, domain-separated hash schemes (e.g. bc-journal-v1 vs bc-experiment-v1 person strings), and forcing one shared primitive across all of them would collapse that intentional separation. Reuse canonicalization/artifact-hashing HELPERS where sensible (tuning.journal.canonical/checksum, experiment.bundle.file_hash/safe_environment) but keep each object's semantic hash DOMAIN distinct -- this is exactly what the implementation below does (own bc-journal-v1-derived checksum via tuning.journal.checksum, not a new global scheme). experiment.bundle.safe_environment() confirmed as a concrete existing reusable input, not something to recreate -- already reused directly in the implementation.

IMPLEMENTED 2026-09-10: tools/bigcherry/environment_inputs.py -- EnvironmentInput (kind: observation/action/artifact, explicit scope: build/run/both/evidence_only per the prior review's projection requirement), EnvironmentInputSet with per-scope digest_for()/build_digest()/run_digest() (evidence_only never composed into either), artifact_input() keying offline tuning files (hipBLASLt/CK/rocblas-gemm-tune) by logical_role + content digest rather than host path, action_input() requiring pre/post observation + restored outcome for reversible mutating actions, secret-pattern rejection reusing experiment.bundle.SECRET. 11 focused unit tests in tools/tests/test_environment_inputs.py, all passing (determinism, secret rejection, path-vs-role-keying, scope partitioning, duplicate-name rejection, restored-outcome enforcement). Wiring into campaign_build.py/campaign_lane.py/artifacts.py/provenance.py (per RE37's original Files list) NOT yet done -- this commit lands the typed model + tests only; integration is the next step whenever this item is picked back up for the campaign-plumbing half of its scope.

## Change Log

- 2026-09-09T10:59:22.190994+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:16:30.032077+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.497934+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:07:29.804131+00:00 (updated-by): Updated: section:notes
- 2026-09-10T00:08:03.286883+00:00 (updated-by): Updated: order=2, priority='P1'
- chg_20260910_000828_reviewed-and-re-planned-all-pe_3612
- 2026-09-10T00:08:28.954888+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.365735+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:19:20.773386+00:00 (updated-by): Updated: section:notes
- chg_20260910_001944_added-a-typed-environment-inpu_2998
- 2026-09-10T00:19:44.473629+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:27:20.419417+00:00 (updated-by): Updated: section:files, section:validation
- 2026-09-10T02:12:40.863964+00:00 (state-transition): State: pending → completed
