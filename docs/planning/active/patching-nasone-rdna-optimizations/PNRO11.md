---
id: PNRO11
order: 0
plan: patching-nasone-rdna-optimizations
state: pending
created-at: '2026-09-09T10:52:56.925530+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P1
---

# Replicate lm_head for DFlash2/DSpark under tensor split

## Description

Replicate lm_head for DFlash2/DSpark full-vocabulary ranking under tensor split, preserving non-replication controls and explicit VRAM limits.

## Steps

- Detect draft architecture/capability from GGUF metadata before target load, not filename.
- Carry explicit output_replicated through model parameters/construction and mark output weight and bias mirrored in Meta split state.
- Enable it only when selector/Markov metadata proves full-output ranking and the draft lacks its own output.weight.
- Cover shared-target lm_head and draft-owned lm_head; the latter must not force target replication.
- Test 2+ GPU tensor split, layer-split, ordinary MTP/Eagle/no-spec controls, candidate selection vs single-GPU/reference, and VRAM fit failure.

## Detailed Solution & Technical Design

Full-vocabulary ranking cannot use a single shard without distributed top-k; this fix chooses replication. Weight and bias mirror states must stay coherent. Replication cost is an operational correctness constraint; distributed top-k is separate future work.

## Code Samples & Guidance



## Files

Speculative model params/load; Meta split-state; DFlash model loading/selector metadata; lm_head weight+bias placement; TP/reference tests; VRAM accounting.

## Validation

Exact candidate selection under TP; full output on every rank; bias placement; no replication when draft owns output; layer/no-spec controls; VRAM delta and fit rejection.

## Effort & Risk



## Standards

Architecture metadata, not names; topology proof; exact selection correctness; resource accounting.

## Acceptance Criteria

Full-vocabulary ranking matches reference under TP; unnecessary replication is absent; weight+bias states are coherent; memory cost is within declared limit or load fails clearly.

## Notes

Supersedes: NRO12
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-nasone-rdna-optimizations-nro12

## Change Log

- 2026-09-09T10:52:56.925530+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:09:25.733744+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.099464+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.751906+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:44:19.935492+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_024433_three-more-nasone-successors-n_7555
- 2026-09-10T02:44:33.164554+00:00 (updated-by): Updated: section:ledger-events
