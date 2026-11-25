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

2026-11-25: the freeze on kernel optimisation is lifted for this item (owner decision); there are no adversaries here, so identity and provenance machinery stays minimal. Approach: build the discrete harness first (RCS25) so each candidate is checked in minutes on one card, then prototype. Candidate that fits the register budget: a taller token tile with a smaller weight tile (for example 64 weight rows by 256 token rows, the same accumulator count as 128 by 128), which halves the decode redundancy at 129 to 256 rows and halves the number of decodes at 2048 (16 bands become 8); a decoded tile shared by two sequential token bands needs both accumulator sets live, which exceeds the 256 register cap. Checks: resource record, output hashes against the shipped build at 129 to 2049 rows including tails, cold and warm replay on both cards (R9700 with the window-agreement rule), then Gate E and the first-token rows at release. GPT reviews the design and returns any code as text.

2026-11-25 result: the 64 weight x 256 activation tile for 129-256 rows was built and gated on gfx1100 with the discrete harness (about 3 minutes a build): byte identical (555 of 555 hashes), no spills, but four forms need 240-246 VGPR (the shipped tile's cap was 239) and 47,616 B of shared memory; cold replay on 72 points shows no win (2-3 percent faster on three shapes within noise, 4-6 percent slower on the largest, 662 to 689 and 665 to 700 microseconds). Halving the weight decodes is paid back by the doubled activation traffic. Archived unchanged (docs/archive/unmerged-candidates-2026-11/vtp58-iqr-tall-tile.patch). Open directions: lower the decode cost per weight itself (the fused single band still pays about 20 percent over int8; the same fused tile is the floor for every band count), or a K-split variant that keeps two bands' decode shared (needs the accumulators in registers for both and does not fit). Harness caveat from the same run: a harness-built gfx1100 libr3 differs from the shipped release in five IQR forms (240 VGPR with 2-3 spills against 236 and none) although the sources compose the same; compare candidates against a harness-built baseline, and find the flag that differs.

2026-11-26 next execution roadmap (docs/planning/NEXT_EXECUTION_ROADMAP.md; main 725bd8951). Assumes native YAML serving cutover and initial min-cap qualification accepted; does not claim they are done. Stage 4 conditional only: 64x256 tall tile rejected (4–6% slower largest shapes); do not rerun. Current IQR GEMM owner bc-iqr since #1603, no new libr3 rewrite. Only lower per-weight decode cost or feasible sharing with bounded traffic/resource proof; test actual unit/row and retain archived candidate. No requirement for every historical item before serving promotion.

2026-11-26 roadmap reconciliation at 165d6f724: Absorb VTP56 residual: single-band fused tile qualified both arch and integrated via bc-iqr; original byte/resource/replay/TTFT gates retained for actually changed cases, no need to repeat shipped forms. Multi-band 64x256 candidate already rejected/archived; only lower per-weight decode cost or feasible sharing if measured exposed, justify remaining spills with resource records. Preserve ragged N, chained/plain, both trellis kinds and unaffected controls for changes. VTP57 owns remaining decode/vocabulary issues.

## Change Log

- 2026-11-25T00:42:24.671410+00:00 (created-by): Created by lead
- 2026-11-25T00:42:47.934646+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:effort_risk, section:acceptance_criteria, section:notes
- 2026-11-25T02:12:40.230589+00:00 (updated-by): Updated: section:notes
- 2026-11-25T04:07:56.842281+00:00 (updated-by): Updated: section:notes
- 2026-11-25T18:52:25.163112+00:00 (updated-by): Updated: section:notes
- 2026-11-25T19:02:06.096040+00:00 (updated-by): Updated: section:notes
