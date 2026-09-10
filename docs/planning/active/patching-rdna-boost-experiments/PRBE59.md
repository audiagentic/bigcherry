---
id: PRBE59
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:57:36.613381+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# FORK-HIP-001: Restore/benchmark rocWMMA FlashAttention on RDNA4

## Description

Research-compare restored rocWMMA FlashAttention on RDNA4 for deep-context workloads; keep conditional and experimental unless stablely superior.

## Steps

Recheck source commit 5aa2f049 and build integration; implement only the low-risk build/kernel selection needed to compare rocWMMA; test R9700 Qwen long-context BF16/Q8 KV at 32K/64K/128K against current FA, decode, gfx1100, and graph on/off controls; run FA backend tests/PPL/long stability; report PP/TG, VRAM/scratch, compile/runtime stability and crash/graph interactions.

## Detailed Solution & Technical Design

Compare rocWMMA FA against current generic/native kernels on selected RDNA4 head/KV shapes rather than automatically reverting upstream decisions. Gate by head size, q_rows, depth, KV type and graph state; retain current kernel fallback.

## Code Samples & Guidance



## Files

HIP FA kernel selection/build integration; backend correctness/PPL tests; deep-context replay and stability evidence.

## Validation

FA tests, PPL, long-run stability; PP/TG, VRAM/scratch, compile/runtime and graph/crash evidence.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Keep only as a conditional experimental path if correctness, stability, and clear deep-context PP benefit are repeatable; otherwise retain current FA path.

## Notes

Supersedes: RD76
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd76

## Change Log

- 2026-09-09T10:57:36.613381+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:14:47.897740+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.393793+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.204866+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:13:33.193064+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_031346_repaired-four-more-migrated-pa_4345
- 2026-09-10T03:13:46.522718+00:00 (updated-by): Updated: section:ledger-events
