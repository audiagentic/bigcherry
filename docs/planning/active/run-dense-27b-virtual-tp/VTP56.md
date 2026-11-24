---
id: VTP56
order: 0
plan: run-dense-27b-virtual-tp
state: pending
created-at: '2026-11-24T02:28:07.143766+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
priority: P2
---

# IQR trellis wide tile (gfx1100): remove the register spill and the per-step shim exchanges from the prefill GEMM of trellis-coded 6/7-bit weights

## Description

**Problem.** In the 2020.8.7 gfx1100 code objects the wide 64-row tile of the IQR weight family spills in every one of its 464 trellis-coded instantiations (table-trellis and computed-trellis forms): 240-256 VGPRs, 6 waves per SIMD, up to 3.6 KB of private memory per thread. It serves rows above 64, i.e. prefill and time to first token for any container whose trunk is trellis-coded (the 27B 6/7-bit IQR conversion in progress). Every instantiation also pays the gfx11 WMMA compatibility shim: 272 lane exchanges and 40 matrix instructions per kernel, converted at every scale group. The decode tiles of the same family are much better: the GEMV tile uses 4-way int8 dot products and no exchanges (at most 24 B spill, in 10 of 1856), and the 16-row tile spills at most 8 B (2 of 364); the computed-trellis 16-row tile carries about 104 exchanges per kernel against 64 for the table rows, because the code expansion adds a conversion.

**Goal.** A prefill GEMM for the trellis forms with no spill and the accumulator kept in the gfx11 layout, converted once per scale group (integer sums are exact, so this is bit-identical by construction), at the same or better occupancy; no change to the on-disk format and none to gfx1201.

## Steps

1. Accumulate-once in the wide tile, the same pattern as the K-parallel int8 tile and the 16-row IQR tile: keep the integer accumulator in the native layout across the 128-K scale-group chain and convert once before the float fold.
2. Spill: find what the 240-256 VGPR allocation holds live (accumulator tiles x token tiles, decoded weight fragments, look-ahead ring) and shrink it: fewer token tiles per wave for the trellis forms, shorter look-ahead, decoded weights staged through the wave's LDS tile rather than held in registers. Select by form kind and arch with a constexpr rule; no model constants.
3. Report waves per SIMD, VGPRs and private bytes for every instantiation before and after; keep a form that loses occupancy only if the timing wins.
4. Also report the computed-trellis 16-row tile's extra exchanges (about 40 per kernel over the table rows) as a candidate for the same change.
5. Dispatch after the drafter IQR tile slice reports, so two sessions do not edit the same kernel source at once.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

- Byte identity against the shipped kernel, one process, full BF16 output bytes: M 65-256, the 27B IQR shapes, ragged N, both trellis kinds, chained and unchained.
- Resource table (VGPR, spill, private bytes, waves) for every instantiation before and after.
- Cold and prefetch-warm replay of the prefill shapes; then time-to-first-token A/B (ABBA, untraced) of the IQR container on the release that carries the change. Regression-only standard.

## Effort & Risk

Medium. The risk is occupancy: shrinking registers can cost a wave; the constexpr rule keeps the old form where it wins.

## Standards



## Acceptance Criteria

- No trellis instantiation of the wide tile spills, or each remaining spill carries bytes and a reason.
- Output bytes identical; no replay or TTFT regression beyond spread; gfx1201 unchanged.

## Notes

**2026-11-24 census.** Static scan of the 2020.8.7 gfx1100 code objects, trellis forms only, exchange instructions per kernel: GEMV tile 0 (dot-product based), 16-row tile 64 (table trellis) and 104 (computed trellis), wide tile 272 (40 matrix instructions), prefill tile 96-100. Related: VTP50 (shim families), VTP37.

## Change Log

- 2026-11-24T02:28:07.143766+00:00 (created-by): Created by agent
- 2026-11-24T02:28:29.229864+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:effort_risk, section:acceptance_criteria, section:notes
