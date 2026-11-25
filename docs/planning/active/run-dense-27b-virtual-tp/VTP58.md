---
id: VTP58
order: 0
plan: run-dense-27b-virtual-tp
state: pending
created-at: '2026-11-25T00:42:24.671410+00:00'
breadth: ''
skill: advanced
created-by: lead
work: L
priority: P2
---

# IQR prefill beyond one band: fused tile for two bands (129 to 256 rows) and lower decode cost per weight

## Description

After the single-band fused tile, the 6-bit prefill is still 1.2 to 1.26 times the int8 route above one band, and every extra 128-row band costs a near-constant pass. Remaining lever: decode into fragments once per block for two bands (129 to 256 rows) and cut the decode cost per weight itself.

## Steps

1. Budget: a two-band block needs the accumulators of 256 rows; the single-band tile is at 236 to 239 VGPR with 36,864 B of shared memory, so H=2 exceeds the 256 register cap unless the tile geometry changes (fewer weight tiles per wave or a shorter lifetime); derive the feasible geometry from the resource records. 2. Candidate designs: a taller token tile with a smaller weight tile, or the decoded tile shared in shared memory across two sequential bands within one block. 3. Gate each: resource table, byte identity at 129 to 256 and the tails, cold and warm replay, controls unchanged. 4. gfx1201 with the warm-up rule (50 or more launches, window agreement). 5. Decode cost per weight: trellis state advance and value computation, packed math.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Resource records of every changed symbol (no spill, no lost wave); byte identity against the shipped plugin at 129 to 2049 rows including tails; cold and warm replay against the shipped route and the int8 route; serving first token at exact prompt token counts of 136 to 256.

## Effort & Risk

Large; the register budget is the main risk (the plane route's gfx1201 variants already spill).

## Standards



## Acceptance Criteria

A candidate that is byte identical, resource clean and faster on every affected point, or documented rejection with the register budget.

## Notes

Measured context: VTP56 notes. Kernel work follows the composition rules (RCS19 to RCS22): additive, suffixed names, anchor gate. GPT is review only on this work.

## Change Log

- 2026-11-25T00:42:24.671410+00:00 (created-by): Created by lead
- 2026-11-25T00:42:47.934646+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:effort_risk, section:acceptance_criteria, section:notes
