---
id: VTP57
order: 0
plan: run-dense-27b-virtual-tp
state: pending
created-at: '2026-11-24T04:47:19.958194+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
priority: P1
---

# IQR decode: residual head costs and per-weight decode efficiency

## Description

**Problem.** The 27B trellis-coded 6/7-bit IQR container (22.5 GiB, 22% smaller than the int8 one) decodes slower than the int8 container on two XTX cards (2020.8.7, step-phase trace, rank 0): the verify pass takes 39.8 ms against 30.9 ms at 4 rows (1 stream), 43.1 against 33.5 at 16 rows, and 74.8 against 36.7 at 24 rows (6 streams): +9 ms/step at 1-4 streams, 2x at 6 streams. At 4 rows the trunk GEMMs take 19.8 ms (computed-trellis tiles 11.4 + gated gate/up form 8.5) against 14.7, and the vocabulary head 5.3 ms against 0.87. At 24 rows the head takes 26.9 ms and the trunk 29.8 against 18.3. Static resource records of the gfx1100 code objects explain the cliff: the head tile spills at every token-tile count (1280, 5576, 8196, 11012 B private at 1-4 tiles, 240-256 VGPRs) and runs two integer matrix passes per 128-K block through the WMMA compatibility shim; 17-32 rows use two tiles, 5x the time for 1.5x the rows. The trunk's 17-64 row path is the wide tile, which spills 3.6 KB in every trellis instantiation; the existing two-slice 16-row route covers only the 4-bit rotated table form.

**Goal.** Decode of the IQR container at or better than the int8 container's speed (fewer bytes should win), with byte-identical outputs.

## Steps

1. Head tile: accumulate the two integer passes (high and low halves of the 16-bit activation) in the gfx11 layout and convert once per block (exact integer sums, bit-identical); remove the spill with a constexpr rule on tile count and arch (for example 16-token slices above 16 rows, re-reading the class weights per slice, if replay shows re-reading is cheaper than the spill).
2. Trunk trellis GEMMs at 17-64 rows: extend the slicing (or cap the wide tile's token tiles) to the computed-trellis and table-trellis forms, keyed on form kind and arch.
3. Trunk trellis decode at 1-16 rows: measure the per-weight ALU cost of the computed trellis against the plain int8 tile at equal bytes (VGPRs, instructions per decoded weight, stream- or ALU-bound) and decide the lever from the numbers.
4. A/B on the serving layout after each step: IQR container against int8 container, ABBA, 1/4/6 streams.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

- Byte identity of outputs (one process, full bytes) at 1-64 rows: head at the four class widths and the rank shares of 2 and 3 ranks; trunk shapes, both trellis kinds, chained and unchained, gated and plain.
- Resource table (VGPR, spill, waves) of every changed instantiation before and after; cold and prefetch-warm replay; then the serving A/B against the int8 container. Regression-only gate for each step.

## Effort & Risk

Medium. Risk is occupancy and re-read traffic when slicing; the form-and-arch rule keeps the old path where it wins.

## Standards



## Acceptance Criteria

- No spill in the head tile at any token-tile count; head at 17-32 rows no more than twice the 16-row time.
- IQR container decode ms/step no slower than the int8 container at 1, 4 and 6 streams on the same layout, or the remaining gap is explained by a measured per-op cause.

## Notes

**2026-11-24 trace.** Step-phase trace (rank 0, two XTX, 2020.8.7). Verify pass ms by rows 4 / 8 / 16 / 24: int8 container 30.9 / 31.1 / 33.5 / 36.7; IQR container 39.8 / 40.0 / 43.1 / 74.8. Drafter passes are identical for both. Decode ms/step in the 12-arm ABBA: 24.39 vs 34.60 (1 stream), 28.04 vs 38.50 (4), 30.92 vs 61.41 (6); first-token latency +10% at 512 tokens, +4% at 2048 (prefill is not the exposed side). Quality (exact wire, spec depth 0): mean KLD 0.001535 vs 0.001979, top-1 97.34% vs 96.99%. Related: VTP56 (wide tile), VTP50 (shim families).

**2026-11-25 status.** Shipped: vocabulary head tile (#1501), IQR accumulate-once (#1505), native fragments (#1507), head cut for three ranks (rad0063/rad0064). IQR container on three ranks, ms/step at 1/6/8 streams: 21.5 / 29.9 / 33.0 against 20.2 / 28.5 / 31.2 for the int8 container (+4..8% time for 21% fewer bytes; before this series 30 / 56-61 / 58-60). Sweep of 2020.9.2 against 2020.8.7 and 2020.11.1 against 2020.11.0 found no regression. Rejected: letting the fused gated fold take 17-32 rows (rad0065, local only): it regressed the serving A/B, so the gated band stays at 16. Residue: the vocabulary-head tile at three and four token tiles still spills a little (52 and 204 bytes private); the three-rank head pads 33.6% extra rows (class re-cut needs a token-probability table). Why six bit is not faster than eight: the computed-trellis tile spends about 10 vector ops per weight after native fragments (about 7.6 in decode proper, the rest in the tile's transpose, packing and fold) against about 11.5 available per weight at the card's bandwidth; removing the hash and byte-sum entirely saves only 10-15%, so the floor is the tile structure, not the code. A transpose-free tile is the open design (deep-speed session).

2026-11-26 roadmap reconciliation at 165d6f724: Narrow to remaining head NT3/NT4 private bytes, padding/row-class cost and actual per-weight decode bottleneck; shipped head, native fragments and chained wide forms in bc-iqr. Use docs/evidence/2026-11-25-iqr-deep-ship and overlay verification. No reimplementation of already gated 44 symbols or archive candidates; byte identity/resource/replay and actual serving only where changed. Keep acceptance explanation of residual IQ8R gap.

## Change Log

- 2026-11-24T04:47:19.958194+00:00 (created-by): Created by agent
- 2026-11-24T04:47:38.589362+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:effort_risk, section:acceptance_criteria, section:notes
- 2026-11-25T19:02:10.841773+00:00 (updated-by): Updated: section:notes, section:title
