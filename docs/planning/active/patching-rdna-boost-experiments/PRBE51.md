---
id: PRBE51
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:57:01.483332+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# UP-HIP-004: Full GPU input-layer offload on AMD UMA

## Description

Qualify full GPU input-layer offload for AMD UMA/iGPU gfx1151, preserving CPU-offload and discrete-GPU behavior.

## Steps

Recheck PR #27426; implement input-layer tensor placement so UMA input tensors remain GPU-resident; add Strix Halo/gfx1151 correctness and synchronization tests; compare full offload against CPU-offload and discrete XTX/R9700 controls across prefill batches/model sizes; report sync count, PP, and residency; keep hardware-scoped fallback.

## Detailed Solution & Technical Design

Avoid repeated CPU/GPU synchronization on AMD UMA by retaining input-layer tensors on the GPU. Gate only integrated gfx1151/UMA hardware and preserve existing policy elsewhere; this is distinct from PRBE17 host-buffer async correctness.

## Code Samples & Guidance



## Files

HIP input placement/offload policy; UMA residency/synchronization tests; Strix Halo replay and discrete/CPU-offload control evidence.

## Validation

Output parity; synchronization count and GPU residency; PP across model/batch controls. Acceptance: hardware-scoped repeatable prefill improvement without discrete-GPU regression.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Promote only for verified UMA/iGPU hardware with output parity, GPU residency, reduced synchronization, and repeatable prefill gain; never generalize to discrete GPUs without independent evidence.

## Notes

Supersedes: RD61
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd61

## Change Log

- 2026-09-09T10:57:01.483332+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:14:15.456287+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.359287+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.144228+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:11:46.041972+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_031217_repaired-four-more-active-succ_7909
- 2026-09-10T03:12:17.932869+00:00 (updated-by): Updated: section:ledger-events
