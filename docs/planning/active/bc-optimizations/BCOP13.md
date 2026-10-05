---
id: BCOP13
order: 13
plan: bc-optimizations
state: pending
created-at: '2026-10-05T04:42:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: M
---

# Land corruption-resistant performance-integrity qualification

## Description

Follow-through for QFP28. External Flash-Next investigation demonstrated that corrupt/skipped MoE work can appear 2-4x faster. BigCherry needs a reusable correctness and physical-work plausibility gate before accepting scheduler/data-movement speedups. QFP28 specifies the design; BCOP13 ensures it is actually implemented and applied.

## Steps

1. Implement the reusable one-process multi-request/multi-ubatch gate under `tools/lab/flash-next/` as specified by QFP28.
2. Prove sensitivity using an injected-corruption fixture or known-bad external gather build.
3. Add expected/observed transfer bytes, implied GB/s and measured link ceiling to data-movement benchmark records.
4. Apply the gate to 1332 and 1295 before either is performance-promoted.
5. If host-expert residency becomes production-relevant, qualify existing staging-ring capacity at equal effective residency; do not create a QFP28-private ring.
6. Feed pass/fail results back into QFP28, QFP22/QFP04 and relevant MET items.

## Related

QFP28, QFP22/1332, QFP04/1295, MET01, MET06.

## Acceptance Criteria

- Gate rejects known corruption and passes a known-good control.
- Multi-request, >1-ubatch and post-abort behavior are covered.
- Data-movement results include physical transport plausibility.
- 1295/1332 throughput cannot be promoted without PASS.
- No duplicate gather/ring/cache/scheduler mechanism is introduced.
