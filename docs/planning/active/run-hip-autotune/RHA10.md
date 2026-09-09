---
id: RHA10
order: 10
plan: run-hip-autotune
state: pending
created-at: '2026-09-09T19:45:36.770283+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
priority: P0
---

# Close production admission with work-equivalence and replay launch proof

## Description

Complete the remaining RHA04 admission gate without conflating diagnostic observations with production timing. Bind stock/native/replay to one source/build identity, prove equivalent workload/correctness, and prove final tuned kernels actually launched in replay.

## Steps

- Extract and verify source/build/generated-input digests for every production arm against one immutable campaign identity.
- Add or consume correctness/work-equivalence evidence for the matched stock/native/replay workload.
- Run a diagnostics-off replay activation capture using the exact production replay binary/cache and retain final_tuned_launches, final_native_launches, fallback and cache identity.
- Join the evidence into a machine-readable admission record and only then set performance_admitted=true where all gates pass; otherwise retain the fail-closed blocker.
- Update RHA04 with the admission decision and preserve all raw artifacts.

## Detailed Solution & Technical Design

Use the maintained server-bench runner and existing execution-audit/admission schemas. Keep replay-diagnostic/framework observations separate from production arms. The minimum positive replay proof is final_tuned_launches > 0 with matching source, cache, signature and hardware identities; cache-loaded alone is insufficient.

## Code Samples & Guidance



## Files

- tools/bigcherry/campaign/benchmark.py
- tools/bigcherry/tuning/execution_audit.py
- docs/reference/testing/TEST.md
- docs/evidence/<rha10-admission-run>/
- docs/planning/active/run-hip-autotune/RHA04.md

## Validation

- All production arms share source/build/model/topology/workload identity.
- Correctness/work-equivalence passes.
- Replay reports final_tuned_launches > 0 and matching cache/signature identity.
- No diagnostics are present in timed arms.
- Admission record is fail-closed and reproducible.

## Effort & Risk

M: evidence and schema integration are bounded, but hardware rerun may be required if the existing raw logs cannot prove final tuned launches.

## Standards

- Capability rebaseline v3 REVIEW_PROTOCOL.md
- docs/reference/testing/TEST.md
- No llama-bench for server-bench qualification.

## Acceptance Criteria



## Notes

RHA04 remains the parity owner; RHA08/RHA09 are dependencies/evidence providers, not admission substitutes.

## Change Log

- 2026-09-09T19:45:36.770283+00:00 (created-by): Created by agent

## Ledger-events

- chg_20260909_194609_separated-completed-parity-cap_7282
- 2026-09-09T19:46:09.213125+00:00 (updated-by): Updated: section:ledger-events
