---
id: PRBE74
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:58:43.924628+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# Q4_K MMVQ wide-load / x4 activation / VDR candidate family (AMD #84 + #88 as one experiment)

## Description

Evaluate the AMD #84/#88 Q4_K MMVQ wide-load/x4 activation/VDR candidate family as one gated experiment with ablations and strict numerical contract.

## Steps

Recheck equivalent plans; implement one experiment with causal knobs: native; x4 layout plus matching consumer; unroll; barrier; VDR=8 ownership; best-plus-nwarps. HIP/Q4_K/MMVQ only with legality and native fallback. Run repeated fixed-seed direct-op correctness across long K before any benchmark on gfx1201 and gfx1100 with unaffected quant control; report candidate and decode evidence.

## Detailed Solution & Technical Design

Evaluate interacting Q4_K optimizations together, never expose an x4 producer to a plain consumer. Require matching block_q8_1 semantics, alignment/contiguity and ncols/decode eligibility, can_execute assertions, and native fallback. Do not assume gfx1151 winner transfers to gfx1100/gfx1201.

## Code Samples & Guidance



## Files

mmvq.cu and vecdotq.cuh candidate code/registration; experiment recipe and qualification artifacts; candidate patch module.

## Validation

Every ablation passes repeated machine-readable max abs/rel direct-op correctness including long K and same-seed repeats before performance; gfx1201 primary/gfx1100 holdout, native/unaffected quant controls; kernel time and decode t/s separately.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Promote only candidates with strict direct-op parity, matching x4 producer/consumer, legal narrow eligibility, and repeatable kernel/E2E evidence; native remains default otherwise.

## Notes

Supersedes: RD98
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd98

## Change Log

- 2026-09-09T10:58:43.924628+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:15:54.994935+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.462729+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.305180+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:20:34.632180+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_032047_repaired-five-more-active-succ_6361
- 2026-09-10T03:20:47.972545+00:00 (updated-by): Updated: section:ledger-events
