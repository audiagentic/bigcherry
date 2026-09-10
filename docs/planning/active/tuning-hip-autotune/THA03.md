---
id: THA03
order: 0
plan: tuning-hip-autotune
state: pending
created-at: '2026-09-09T10:48:17.379394+00:00'
breadth: ''
skill: intermediate
created-by: capability-rebaseline-v3
work: S
priority: null
---

# Harden signature coverage for RD12 dst_gate and RD17 x_scale_channel_dst fusion modes

## Description

Harden signature coverage for experimental PRBE11 dst_gate and PRBE14/RD17 x_scale_channel_dst modes without blocking the upstream GLU harness.

## Steps

When PRBE11 moves toward production, add explicit patch-conditional dst_gate signature flag instead of GATE+glu_op NONE inference. When PRBE14 moves toward production, add x_scale_channel_dst semantic flag/field and destination-channel scale length (dst.ne[1]); add signature mapping and backward-compatibility tests. Keep both modes out of THA02 dependency until promoted.

## Detailed Solution & Technical Design

Record semantics that are currently implicit in experimental signatures: RD12 dst_gate fused mode and RD17 channel-indexed x_scale. Use existing patch-conditional signature-field patterns and fail closed for unknown modes.

## Code Samples & Guidance



## Files

hip-autotune signature types/serialization/mapping; PRBE11/PRBE14 conditional fields; signature and evidence tests.

## Validation

Synthetic and production-shaped signature round trips; distinguish dst_gate from GLU NONE; reconstruct x_scale_channel_dst length/indexing; ensure THA02 upstream GLU unaffected.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Experimental signatures remain unambiguous and round-trip semantic fields when promoted; no change to THA02 upstream-GLU scope or default behavior.

## Notes

Supersedes: HI120
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-hip-autotune-hi120

## Change Log

- 2026-09-09T10:48:17.379394+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:04:20.078449+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.800282+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.278902+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:25:44.385035+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_032600_repaired-two-more-tuning-succe_5113
- 2026-09-10T03:26:00.704225+00:00 (updated-by): Updated: section:ledger-events
