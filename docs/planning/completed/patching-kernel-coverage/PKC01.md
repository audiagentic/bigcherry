---
id: PKC01
order: 0
plan: patching-kernel-coverage
state: completed
created-at: '2026-09-09T10:51:42.597496+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: S
priority: P2
---

# Decide whether non-GEMM lifecycle representation has a gap

## Description

Decision gate completed by checking EC16's orthogonal target classification and EC19's computed lifecycle status against the representative non-GEMM owners. FLASH_ATTN (RD04-RD06), GDN/SSM (RD50, with RD51-RD53 subsumed), graph fusion, orchestration, and TP topology all have existing target-kind representations; no lifecycle or identity gap was found.

## Steps

1. Implement the still-valid future scope.
2. Run the stated acceptance and evidence gates.
3. Preserve predecessor provenance and record successor evidence under this ID.

## Detailed Solution & Technical Design

Reuse EC16/EC19. Keep hypothesis.family limited to the five runtime matmul families; use target.kind for attention, ssm_gdn, graph_fusion, orchestration, tp_topology, and other non-GEMM classifications. Lifecycle status remains EC19's computed concern, while correctness/provider/performance evidence remains with the owning RD/RO item. Do not create a parallel registry or result-key system.

## Code Samples & Guidance



## Files

- successor-specs/patching-kernel-coverage-kc01.md
- docs/evidence/2026-09-10-pkc01-representation-review/README.md
- config/experiment-contracts.toml
- tools/bigcherry/experiment/contract.py

## Validation

Field-by-field matrix and ownership decision recorded in docs/evidence/2026-09-10-pkc01-representation-review/README.md. Verified current contracts for RD04/RD05/RD06 (target.kind=attention), RD50 (target.kind=ssm_gdn), RD13 (graph_fusion), RD19/RD39-44 (orchestration), RD20 (tp_topology), and RD08 (kernel_family). No code change or duplicate registry was required.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

The field-coverage matrix proves existing EC16/EC19 representation is sufficient for FLASH_ATTN and GDN; ownership is recorded; no duplicate registry, result key, or lifecycle implementation is created.

## Notes

Supersedes: KC01
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-kernel-coverage-kc01

2026-09-10: PKC01 decision is no gap. EC16 already supplies the required orthogonal target kinds and EC19 remains the computed lifecycle owner. Evidence: docs/evidence/2026-09-10-pkc01-representation-review/README.md.

## Change Log

- 2026-09-09T10:51:42.597496+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:08:08.830735+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.035040+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T15:21:10.309548+00:00 (state-transition): State: pending → in_progress
- 2026-09-09T15:21:25.894827+00:00 (updated-by): Updated: section:description, section:detailed_solution, section:files, section:validation, section:acceptance_criteria, section:notes
- chg_20260909_152204_closed-pkc01-existing-experim_8209
- 2026-09-09T15:22:04.163091+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T15:22:11.442174+00:00 (state-transition): State: in_progress → completed
