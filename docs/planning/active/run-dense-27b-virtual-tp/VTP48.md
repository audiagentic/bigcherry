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

The K-parallel int8 tile pays the gfx1100 matrix-multiply shim (lane exchanges for both fragments, and the accumulator) at every K step; accumulate-once removes the accumulator exchange at two or more token tiles but the fragment exchanges remain. A path with no matrix unit at all would remove the rest for the 1-8 row decode rows. gfx1100 has the packed int8 dot instruction (`__builtin_amdgcn_sudot4`: four int8 products accumulated into int32 per lane); no gemm path in libr3 or libr4d uses it today. Expected: up to about 0.5 ms a step at one stream (M 1-4), none at 6-8 streams (M 24-32); an early NO-GO is likely above M 5.

**What we will try.** A scratch variant of the int8 tile for M <= 8: weights as packed int8x4 words in registers, activation rows broadcast from LDS, one int8 dot per four K per row, int32 group sums kept exact and scales applied once per scale group exactly as the shipped kernel, so output is byte-identical by construction. Compute check, not a measurement: at 960 GB/s a row needs about 1.9 TOPS of int8 work per weight stream, so M = 8 is about 15 TOPS against a vector dot peak near 120 TOPS; the limit is instruction overhead (broadcast, per-row accumulators, scale application).

## Steps

1. Scratch single-unit variant for M <= 8, byte identity against the shipped row.
2. Cold timing replay at the six rank-0 shapes for M 1, 4, 8, 16 against the shipped tile; find the crossover M, if any.
3. If a crossover exists above M 5, add a row to libr3 with a selection rule keyed on M and N, then decode ms/step ABBA at 1 and 4 streams.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

- int32 sums and outputs byte-identical to the shipped row at every tested M.
- Resource record without spills.
- Serving ABBA at 1 and 4 streams if a row is added.

## Effort & Risk

Medium; an early NO-GO is likely if the dot path is instruction-bound above M 4.

## Standards



## Acceptance Criteria

- A measured crossover at M >= 6 with at least 10% lower isolated time at one rank-0 shape, then a serving gain above the seed spread; otherwise a NO-GO recorded with the replay table.

## Notes

Pending: scheduled after VTP47 step 1, which uses the same scratch harness.

Withdrawn: not justified by our own measurements.

Reinstated 2026-11-24.



Ceiling basis (2026-11-24): the 0.5 ms at one stream is a compute-feasibility model (operations per weight byte against the vector dot rate), not a measurement and not a tape sum; it is a hypothesis until the scratch replay exists. Downgraded: no brief before the replay shows a crossover.



**2026-11-24 ACCEPT-WORK (exploratory).** No regression is expected from adding a row for M <= 8 that is selected only where the replay shows it faster; a hypothesis until the scratch replay exists. The earlier 'likely NO-GO above M 5' stays a prediction, not a gate.

## Change Log

- 2026-11-23T20:42:25.061342+00:00 (created-by): Created by agent
- 2026-11-23T20:43:16.966373+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:effort_risk, section:acceptance_criteria
- 2026-11-23T20:54:44.243191+00:00 (updated-by): Updated: section:description, section:steps, section:notes
- 2026-11-23T21:10:56.920226+00:00 (updated-by): Updated: section:notes
- 2026-11-23T21:11:12.715307+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:effort_risk, section:acceptance_criteria
- 2026-11-23T21:11:22.195660+00:00 (state-transition): State: pending → deprecated
- 2026-11-23T21:14:32.596842+00:00 (state-transition): State: deprecated → pending
- 2026-11-23T21:15:08.429046+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:effort_risk, section:acceptance_criteria, section:notes
- 2026-11-23T22:51:40.135239+00:00 (updated-by): Updated: section:notes
- 2026-11-23T23:21:26.385098+00:00 (updated-by): Updated: section:notes
