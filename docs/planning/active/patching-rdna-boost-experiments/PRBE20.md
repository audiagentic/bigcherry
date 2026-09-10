---
id: PRBE20
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:54:49.394058+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# Make decode and speculative-verify batches bit-identical (attention + mmvq + CPU cluster)

## Description

Complete the coordinated decode/speculative-verify bit-identity cluster. Standalone subset exists; attention and post-fix MMVQ portions remain composition-gated.

## Steps

- Treat all five source commits as one determinism property: flash attention, non-flash attention, CPU, RDNA4 MMVQ/SSM and RDNA3 MMVQ.
- Keep standalone materialized subset explicit; port fattn portions only after PRBE02/03/06 retained and in the required family order.
- Port RDNA4/RDNA3 portions from branch-tip post-fix state under PRBE19; do not create half-deterministic composition.
- Run decode vs speculative-verify on identical inputs and compare byte identity across reachable gfx1100/RDNA4 paths and CPU cluster.
- Preserve native controls and record any determinism cost; no performance promotion is implied.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

attention fattn sources; ggml-cuda.cu; mmvq.cu; CPU sgemm; PRBE19 post-fix regions; patch 1210; exact decode/verify fixtures and campaign evidence.

## Validation

All five commit identities; dependency/order checks; byte-identical outputs; native control; gfx1100 and reachable RDNA4/RDNA3; graph/capture where applicable; no unintended performance regression.

## Effort & Risk



## Standards

Coordinated determinism change; no half-cluster acceptance; branch-tip post-fix provenance.

## Acceptance Criteria

Full cluster, not only standalone subset, produces byte-identical decode/verify outputs under declared paths; composition dependencies are satisfied; partial ports are not promoted.

## Notes

Supersedes: RD26
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd26

## Change Log

- 2026-09-09T10:54:49.394058+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:12:02.133012+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.219018+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.935699+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:51:52.755045+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_025218_rdna-successors-prbe2022-now_5714
- 2026-09-10T02:52:18.396789+00:00 (updated-by): Updated: section:ledger-events
