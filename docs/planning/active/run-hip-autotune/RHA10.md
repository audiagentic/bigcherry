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

Close the remaining production admission gate with source/work-equivalence and a two-build replay proof. Replay activation is observed in a compile-time diagnostics build; production timing remains diagnostics-off and uses the matching production replay binary/cache.

## Steps

- Extract and verify source/build/generated-input digests for stock/native/replay production arms and the replay-diagnostics activation arm against one immutable campaign identity.
- Add or consume correctness/work-equivalence evidence for the matched stock/native/replay workload.
- Run or retain a replay-diagnostics activation capture using the exact source/catalog/cache identity; require positive exact hits, zero misses/incompatibilities, and positive non-native/tuned hit rows where the workload reaches them.
- Run the production replay binary with diagnostics disabled under the same workload/topology/cache identity; retain timing and clean teardown separately. Do not require counters that are compiled out of this binary.
- Join the evidence into a machine-readable admission record and only then set performance_admitted=true where all gates pass; otherwise retain the fail-closed blocker.
- Update RHA04 with the admission decision and preserve all raw artifacts.

## Detailed Solution & Technical Design

Use the maintained server-bench runner and existing execution-audit/admission schemas. Keep replay-diagnostic/framework observations separate from production arms. Because patch 0810 compiles the hit recorder only when GGML_HIP_REPLAY_DIAGNOSTICS is enabled, final tuned-launch evidence must come from the identity-matched diagnostic build; production replay timing must come from the diagnostics-off build. Cache-loaded alone remains insufficient.

## Code Samples & Guidance



## Files

- tools/bigcherry/campaign/benchmark.py
- tools/bigcherry/tuning/execution_audit.py
- docs/reference/testing/TEST.md
- docs/evidence/<rha10-admission-run>/
- docs/planning/active/run-hip-autotune/RHA04.md

## Validation

- Exact-source diagnostics witness records final_tuned_launches > 0 after revalidation.
- Diagnostics-off production timing is retained separately with clean teardown.
- Stock/native/replay fixed-seed probe is retained; any mismatch blocks admission until RHA11 resolves it.
- Correctness/work-equivalence passes across the selected corpus.
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

Checkpoint evidence is retained under docs/evidence/2026-09-10-rha10-admission-gate. It proves diagnostic replay activation (54 winners, 9 exact hits, 100% measured coverage, 12 non-native hit rows) but deliberately does not satisfy production admission. A diagnostics-off final_tuned_launches proof and work-equivalence remain required.

Checkpoint evidence is retained under docs/evidence/2026-09-10-rha10-admission-gate. The existing RHA08 checkpoint satisfies the diagnostic activation shape, but its source/build identity must be joined explicitly to the RHA04 production replay binary before admission.

The exact-source diagnostic activation gate is now evidenced: 21,566/21,566 coverage and 13,342 final tuned launches, paired with diagnostics-off production timing. However, the three-prompt deterministic probe found replay divergence on one prompt; RHA11 owns the blocking correctness investigation. RHA10 remains in_progress and performance_admitted=false.

Exact-source diagnostic activation and diagnostics-off production timing are complete, but the three-prompt correctness gate found replay divergence. RHA11 has now isolated two admitted MMVQ winners and generated a corrected cache with both seeded to native; RHA10 remains fail-closed until that corrected cache is installed in the canonical campaign location and activation/timing are rerun.

RHA11 correctness issue is resolved in evidence: both implicated MMVQ winners were seeded to mmvq:native:v1 through the normal exporter, the corrected cache is now installed at the canonical campaign path (original retained as dispatch.cache.pre-rha11), the three-prompt corpus matches stock/native, diagnostics activation returns 0 with 21,566/21,566 coverage and 14,622 final tuned launches, and diagnostics-off production timing returns 0 (pp512 930.03, pp2048 1282.63, tg128 33.83, tg512 34.17). RHA10 remains in_progress/fail-closed pending admission-record refresh and explicit treatment of the 24 replay misses/native fallbacks under the stated zero-miss policy.

## Change Log

- 2026-09-09T19:45:36.770283+00:00 (created-by): Created by agent

## Ledger-events


- chg_20260909_194609_separated-completed-parity-cap_7282
- 2026-09-09T19:46:09.213125+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T19:47:48.323863+00:00 (updated-by): Updated: section:validation, section:notes
- chg_20260909_194753_captured-positive-diagnostic-r_4130
- 2026-09-09T19:47:53.704763+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T19:49:35.186489+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:validation, section:notes
- 2026-09-09T20:09:23.176448+00:00 (updated-by): Updated: section:validation, section:notes
- chg_20260909_200933_the-tuned-path-now-proves-actu_7816
- 2026-09-09T20:09:33.369698+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T20:39:10.108454+00:00 (updated-by): Updated: section:notes
- chg_20260909_203929_the-replay-mismatch-is-now-iso_7697
- 2026-09-09T20:39:29.251175+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T20:45:00.888899+00:00 (updated-by): Updated: section:notes
- chg_20260909_204705_the-quarantined-cache-is-now-t_2378
- 2026-09-09T20:47:05.941088+00:00 (updated-by): Updated: section:ledger-events
