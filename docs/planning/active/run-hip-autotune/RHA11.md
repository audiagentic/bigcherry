---
id: RHA11
order: 11
plan: run-hip-autotune
state: in_progress
created-at: '2026-09-09T20:09:03.181768+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
priority: P0
---

# Resolve replay correctness divergence before production admission

## Description

Investigate and resolve the deterministic output difference observed between stock/native and replay on the exact-source dual-XTX Qwen3.8-27B workload.

## Steps

- Reproduce the mismatched prompt with stock, native and replay using fixed seed, temperature, model, topology and cache.\n- Capture per-signature replay-hit diagnostics using the exact production tensor-split server arguments.\n- Seed both implicated dispatches to mmvq:native:v1 through the normal replay-cache exporter; do not hand-edit runtime metadata.\n- Verify the corrected cache against the three-prompt deterministic corpus.\n- Regenerate the campaign cache in its canonical remote location and repeat RHA10 activation/timing before production admission.

## Detailed Solution & Technical Design

Use the maintained server runner for timing and the existing correctness/evidence tooling for semantic validation. Keep diagnostics-on coverage and diagnostics-off timing separate. The first isolation contrast should be replay with the mismatched signature falling back to native, then replay with the candidate forced only if eligibility is proven.

## Code Samples & Guidance



## Files

- tools/bigcherry/tuning/execution_audit.py
- tools/bigcherry/campaign/benchmark.py
- docs/reference/testing/TEST.md
- docs/evidence/<rha11-correctness-run>/
- docs/planning/active/run-hip-autotune/RHA10.md

## Validation

- Exact production-argument three-prompt corpus reproduces the mismatch before quarantine.\n- Both implicated dispatches are explicitly identified from replay-hit diagnostics.\n- A normal replay-cache export with manifest-bound native seed overrides produces a corrected cache.\n- Corrected replay matches stock/native content on all three prompts.\n- Remaining: install/regenerate the campaign cache, rerun diagnostics activation and production timing, and only then release the RHA10/RHA04 gates.

## Effort & Risk

M: likely requires a bounded candidate isolation run and possibly cache regeneration; correctness risk is release-blocking.

## Standards

- Capability rebaseline v3 REVIEW_PROTOCOL.md
- docs/reference/testing/TEST.md
- No llama-bench for server-bench qualification.

## Acceptance Criteria



## Notes

RHA11 is a prerequisite to closing RHA10 and RHA04. Do not reinterpret the observed mismatch as a performance result.

RHA11 investigation reproduced the RHA10 divergence with exact production server arguments, seed 42, temperature 0, and the three-prompt corpus. Diagnostics hit log identified two admitted MMVQ winners on the failing prompt: dispatch f21133b3ee604fd361cb7eb5ccfd1fc3/signature 19a6ba40abbf5e14029d25cf67782eaa -> mmvq:q8_0:w1:nw4:rpb4:sk1:v1, and dispatch eb625b97022490d8e6ad5dcaa5cba1b9/signature a03e95a3b8808c2ebd4160067f73f01e -> mmvq:q8_0:w2:nw4:rpb4:sk1:v1. Re-exporting with both dispatches seeded to mmvq:native:v1 produced a manifest-valid corrected cache; replay with that cache matched stock/native on all three prompts. This is candidate quarantine proof, not production admission: regenerate the campaign cache from the seed decision, rerun diagnostics activation and diagnostics-off timing, then update RHA10/RHA04. Evidence: docs/evidence/2026-09-10-rha11-correctness/.

## Change Log

- 2026-09-09T20:09:03.181768+00:00 (created-by): Created by agent

## Ledger-events


- chg_20260909_200933_the-tuned-path-now-proves-actu_7816
- 2026-09-09T20:09:33.391094+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T20:38:16.733081+00:00 (updated-by): Updated: section:steps, section:validation, section:notes
- 2026-09-09T20:38:25.814080+00:00 (state-transition): State: pending → in_progress
- chg_20260909_203929_the-replay-mismatch-is-now-iso_7697
- 2026-09-09T20:39:29.272562+00:00 (updated-by): Updated: section:ledger-events
