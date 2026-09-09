---
id: RHA11
order: 11
plan: run-hip-autotune
state: pending
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

- Reproduce the mismatched prompt with stock, native and replay using fixed seed, temperature, model, topology and cache.
- Run replay with the implicated signature/candidate disabled or forced native to isolate whether a tuned kernel causes the divergence.
- Collect token/logit or correctness-harness evidence, not only aggregate throughput; compare against an explicit tolerance contract.
- Quarantine or demote any candidate that fails correctness, rebuild the cache if required, and repeat the production timing/activation evidence.
- Update RHA10/RHA04 only after the replay path passes the chosen correctness gate.

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

- The mismatched prompt is reproduced or explicitly invalidated with provenance evidence.
- Stock/native/replay pass the correctness contract across the selected corpus.
- No candidate remains admitted if it changes output beyond the contract.
- Updated diagnostics coverage still records positive final tuned launches after any quarantine.
- RHA10 remains fail-closed until all checks pass.

## Effort & Risk

M: likely requires a bounded candidate isolation run and possibly cache regeneration; correctness risk is release-blocking.

## Standards

- Capability rebaseline v3 REVIEW_PROTOCOL.md
- docs/reference/testing/TEST.md
- No llama-bench for server-bench qualification.

## Acceptance Criteria



## Notes

RHA11 is a prerequisite to closing RHA10 and RHA04. Do not reinterpret the observed mismatch as a performance result.

## Change Log

- 2026-09-09T20:09:03.181768+00:00 (created-by): Created by agent

## Ledger-events

- chg_20260909_200933_the-tuned-path-now-proves-actu_7816
- 2026-09-09T20:09:33.391094+00:00 (updated-by): Updated: section:ledger-events
