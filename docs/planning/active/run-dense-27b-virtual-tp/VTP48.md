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

The K-parallel int8 tile pays the gfx1100 matrix-multiply shim (lane exchanges for both fragments and the accumulator) at every K step; the accumulate-once change removes most of it at two or more token tiles, but a path with no matrix unit at all would remove the remainder for the 5-8 row decode rows. gfx1100 has the packed int8 dot instruction (`__builtin_amdgcn_sudot4`, four int8 products accumulated into int32 per lane) and a bf16 pair dot; no gemm path in libr3 or libr4d uses the int8 dot today (the float pair dot appears only in attention softmax sums).

**What we will try.** A scratch variant of the int8 tile for M <= 8 that keeps weights as packed int8x4 words in registers, broadcasts the activation rows from LDS, issues one int8 dot per four K per row, keeps the int32 group sums exact and applies the scales once per scale group exactly as the shipped kernel does, so the output is byte-identical by construction. Compute check, not a measurement: at 960 GB/s a row costs about 1.9 TOPS of int8 work per weight stream, so M = 8 is about 15 TOPS against a vector dot peak near 120 TOPS; the limit is instruction overhead (broadcast, per-row accumulators, scale application), which the replay will show.

## Steps

1. Scratch single-unit variant for M <= 8 as described; byte identity against the shipped row.
2. Cold timing replay at the six rank-0 shapes for M 1, 4, 8 and 16 against the shipped tile; find the crossover M, if any.
3. If a crossover exists above M 5, add the row to libr3 with a selection rule keyed on M and N, then decode ms/step ABBA at 5-8 streams.

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

Pending: scheduled after VTP47 step 1, which uses the same scratch harness.

## Change Log

- 2026-11-23T20:42:25.061342+00:00 (created-by): Created by agent
- 2026-11-23T20:43:16.966373+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:effort_risk, section:acceptance_criteria
- 2026-11-23T20:54:44.243191+00:00 (updated-by): Updated: section:description, section:steps, section:notes
