---
id: PRBE10
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:54:10.807250+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# Evaluate gating MUL direct Q8_1 production

## Description

Evaluate gating-MUL direct Q8_1 production only after PRBE05/PRBE09 cache and Q8 contracts are proven; this remains Wave-2 dependent work.

## Steps

- Treat PRBE05 cache correctness, generation lifecycle and capture gates as prerequisites.
- Port only the gating-MUL direct Q8_1 producer at its exact native seam; do not alter unrelated dispatch or fusion.
- Compare baseline plus PRBE05 against baseline plus PRBE05 plus PRBE10 for marginal launch, memory and latency effects.
- Cover qualifying/nonqualifying shapes, graph/non-graph, exact Q8 block identity and native fallback.
- Promote only with statistically supported end-to-end benefit and no numerical, memory or fallback regression.

## Detailed Solution & Technical Design

Direct production must use the proven cache reserve/publish contract and preserve standalone quantization as reference. Unsupported or exhausted paths remain on the native implementation; no default enablement before causal evidence.

## Code Samples & Guidance



## Files

Gating-MUL Q8_1 producer seam; PRBE05 cache API; Q8 reference fixtures; graph/non-graph and causal campaign artifacts.

## Validation

PRBE05 prerequisite; exact Q8 block identity; numerical output; shape/fallback matrix; graph capture; launch/memory counters; causal performance.

## Effort & Risk



## Standards

Dependent causal arm; exact output evidence; fail closed on cache and numerical uncertainty.

## Acceptance Criteria

No implementation is accepted before PRBE05 passes; qualifying outputs match reference and unsupported paths fall back; only a proven marginal improvement without regressions can promote.

## Notes

Supersedes: RD11
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd11

## Change Log

- 2026-09-09T10:54:10.807250+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:11:18.912368+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.173928+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.867375+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:35:17.696577+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_023529_three-more-rdna-successors-now_3176
- 2026-09-10T02:35:29.343335+00:00 (updated-by): Updated: section:ledger-events
