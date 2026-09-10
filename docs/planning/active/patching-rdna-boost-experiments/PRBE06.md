---
id: PRBE06
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:53:51.286520+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# Evaluate RMS norm direct Q8_1 production

## Description

Evaluate direct Q8_1 production in RMS norm only after PRBE05's cache contract and correctness gates are proven.

## Steps

- Prove PRBE05 cache correctness and capture lifecycle first; do not evaluate this child against raw baseline.
- Port direct RMS-norm Q8_1 production with explicit source and format identity, preserving the standalone quantizer/reference path.
- Compare B+PRBE05 against B+PRBE05+PRBE06 so the marginal effect is causal.
- Cover qualifying and nonqualifying patterns, graph/non-graph, numerical output and Q8 block identity, including native fallback.
- Measure quantization launches, memory and graph effects with balanced repeats and record rejection if the marginal result is not positive.

## Detailed Solution & Technical Design

PRBE06 is a dependent producer optimization, not a replacement cache design. Direct output must be equivalent to the registered Q8_1 format and must publish only after enqueue; all unsupported shapes use the existing native path.

## Code Samples & Guidance



## Files

RMS-norm HIP producer seam; PRBE05 cache API; Q8 block/reference fixtures; graph/non-graph campaign artifacts and marginal evidence.

## Validation

PRBE05 prerequisite; numerical equality; exact Q8 block identity; graph/non-graph; fallback patterns; launch/memory accounting; causal B+PRBE05 vs B+PRBE05+PRBE06 performance.

## Effort & Risk



## Standards

Dependent causal arm; fallback preservation; exact output evidence; fail closed on cache or numerical uncertainty.

## Acceptance Criteria

No direct producer is enabled before PRBE05 passes; all qualifying outputs match reference and unsupported patterns fall back; only a statistically supported marginal improvement without quality or memory regression can be promoted.

## Notes

Supersedes: RD10
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd10

## Change Log

- 2026-09-09T10:53:51.286520+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:10:56.565121+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.152218+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.829930+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:32:14.424353+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_023232_the-next-three-rdna-successors_5807
- 2026-09-10T02:32:32.993441+00:00 (updated-by): Updated: section:ledger-events
