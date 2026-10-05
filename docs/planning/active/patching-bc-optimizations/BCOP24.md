---
id: BCOP24
order: 24
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-05T06:14:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: S
---

# Track AMD Q2_K spill and Q1_0 native-permute qualification

## Description

Action/disposition record for the AMD quant-kernel audit captured in BCOP14. BCOP14 contains the historical audit/evidence, but neither BCOP item owns a new production dispatch mechanism. Accepted source must remain exact-upstream or move under the existing HIP kernel/autotune ownership.

## Actions

1. Qualify llama.cpp #29910 Q2_K spill/unroll changes independently on gfx1100/gfx1201 with correctness, resource/ISA and representative ubatch/PP/TG evidence.
2. Qualify #29927 Q1_0 native-permute semantics with the exhaustive host selector check, then MMQ/MMVQ correctness and gfx1100/gfx1201 performance.
3. Classify each candidate independently as `adopt-upstream`, `wait-upstream`, `reject`, or `already-upstream`.
4. If an architecture crossover is proven, route it to existing HIP architecture/autotune ownership; do not create BCOP dispatch policy.
5. Close when both candidates have terminal dispositions or arrive through a validated normal llama.cpp pin.

## Gate

Compiler spill removal alone is insufficient. Promotion requires correctness plus local RDNA3/RDNA4 end-to-end evidence. Disposable qualification patches are removed once upstreamed/rejected.

## Related

BCOP14 (historical detailed audit/evidence); llama.cpp #29910/#29927 lineage; existing HIP kernel/autotune ownership.
