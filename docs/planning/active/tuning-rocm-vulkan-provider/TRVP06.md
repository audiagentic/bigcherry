---
id: TRVP06
order: 0
plan: tuning-rocm-vulkan-provider
state: pending
created-at: '2026-09-09T11:00:36.104225+00:00'
breadth: ''
skill: intermediate
created-by: capability-rebaseline-v3
work: M
priority: null
---

# hipBLASLt independent validation oracle

## Description

Add an independent hipblaslt-bench/provider-log validation oracle for explicit and auto candidates, proving descriptor translation, coverage, winner identity, workspace, and timing without making the oracle a production dependency.

## Steps

1. Add a provider-validate command/tooling path that compares BigCherry measurements with hipblaslt-bench and HIPBLASLT_LOG_MASK=32 observations. 2. Persist exact benchmark binary/version/hash, device fingerprint, dimensions, layouts, transposes, dtypes, candidate/solution identity, workspace, and timings. 3. Compare descriptor representation, candidate coverage, winning index, workspace, and timing within declared tolerances for representative and exhaustive signatures. 4. Classify every discrepancy as missing-provider coverage, translation/representation mismatch, identity/fingerprint mismatch, or expected measurement variance; never silently tolerate it. 5. Keep hipblaslt-bench validation-only and remove any production dependency on HIPBLASLT_TUNING_OVERRIDE_FILE. 6. Gate TRVP04/TRVP05 qualification on a passing oracle report and retain evidence under this item.

## Detailed Solution & Technical Design

The oracle invokes hipblaslt-bench and captures HIPBLASLT_LOG_MASK=32 output outside the runtime path, normalizes both representations into a canonical problem/candidate record, and compares coverage, solution/index, workspace, descriptors, and timing. A provider-legal candidate absent from BigCherry is a completeness bug; a mismatched descriptor or layout/dtype/transpose is a translation bug; fingerprint differences are an identity bug; timing-only deviations use explicit tolerance and remain evidence, not silent success. The command must be reproducible from recorded binary hashes, exact problem signatures, and device/stack fingerprints.

## Code Samples & Guidance



## Files

tools/bigcherry/tuning/provider_validation.py; tuning CLI/provider-validate command; canonical comparison schema and parsers; validation/oracle tests; qualification evidence reports.

## Validation

Run representative and exhaustive signatures through BigCherry and hipblaslt-bench. Confirm matching problem descriptors, candidate coverage, winning indexes/solutions, workspace, and provider logs within declared timing tolerance. Inject missing candidates and descriptor mismatches and verify classified failures. Confirm reports include benchmark binary/version/hash, exact dimensions/layout/transposes/dtypes, solution/workspace, and device fingerprint. Verify no production path imports or requires HIPBLASLT_TUNING_OVERRIDE_FILE and that TRVP04/TRVP05 cannot be promoted without a passing oracle report.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

The independent oracle is validation-only and reproducible. Descriptor and candidate coverage comparisons pass for representative/exhaustive signatures, with all discrepancies classified and actionable. Reports persist exact binary, problem, solution, workspace, timing, and device/stack provenance. No silent tolerance or production dependency on HIPBLASLT_TUNING_OVERRIDE_FILE; TRVP04/TRVP05 qualification is gated on oracle success.

## Notes

Supersedes: RO11
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-rocm-vulkan-provider-ro11

Supersedes RO11. Preserve the independent-gate relationship to TRVP04 and TRVP05 and do not treat hipblaslt-bench as a runtime dependency.

## Change Log

- 2026-09-09T11:00:36.104225+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:17:27.734789+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.556160+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:47.397837+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:42:40.979167+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria, section:notes
- chg_20260910_034253_repaired-trvp04-06-so-the-acti_1663
- 2026-09-10T03:42:53.412159+00:00 (updated-by): Updated: section:ledger-events
