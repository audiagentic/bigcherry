---
id: VTP48
order: 0
plan: run-dense-27b-virtual-tp
state: pending
created-at: '2026-11-23T20:42:25.061342+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
priority: P2
---

# Shim-free int8 dot-product (sudot4) path for 5-8 row decode rows on gfx1100: measure against the K-parallel WMMA tile

## Description

Source: the same reference decode set runs its single-row W4A16 decode GEMV without WMMA, on VALU dot instructions. gfx1100 has the int8 dot (v_dot4_i32_iu8, `__builtin_amdgcn_sudot4`) and bf16 dot2; no libr3/libr4d gemm path uses sudot4 today (only fdot2 in attention softmax sums). The K-parallel int8 tile pays the gfx11 WMMA shim (lane exchanges for both fragments and the accumulator) at every K step; the accumulate-once change removes most of it for nt >= 2, but a shim-free path would remove the remainder for the 5-8 row decode rows. Compute check, not a measurement: at 960 GB/s a row costs about 1.9 TOPS of int8 work per weight stream, M = 8 is about 15 TOPS against a VALU dot peak near 120 TOPS, so the VALU is not the limit in principle; instruction overhead (activation broadcast, per-row accumulators, scale application) is. Hypothesis only.

## Steps

1. Scratch single-unit variant of the int8 tile for M <= 8: weights in registers as packed int8x4, activation rows broadcast from LDS or SGPR pairs, one sudot4 per 4 K per row, group sums kept in int32 and scaled once per scale group exactly as the shipped kernel (bit-identical by construction).
2. Cold timing replay at the six rank-0 shapes at M 1, 4, 8, 16 against the shipped K-parallel tile; find the crossover M, if any.
3. If a crossover exists above M 5, add the row to libr3 with a selection rule keyed on M and N (no model constants), then decode ms/step ABBA at 5-8 streams.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

- int32 sums and outputs byte-identical to the shipped row at every tested M.
- Resource record without spills.
- Serving ABBA at 6 and 8 streams if a row is added.

## Effort & Risk

Medium; an early NO-GO is likely if the VALU path is instruction-bound above M 4.

## Standards



## Acceptance Criteria

- A measured crossover at M >= 6 with at least 10% lower isolated time at one rank-0 shape, then a serving gain above the seed spread; otherwise a NO-GO recorded with the replay table.

## Notes



## Change Log

- 2026-11-23T20:42:25.061342+00:00 (created-by): Created by agent
- 2026-11-23T20:43:16.966373+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:effort_risk, section:acceptance_criteria
