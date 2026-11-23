---
id: VTP46
order: 0
plan: run-dense-27b-virtual-tp
state: pending
created-at: '2026-11-23T20:42:14.800583+00:00'
breadth: ''
skill: advanced
created-by: agent
work: L
priority: P1
---

# Decode attention (h256 gqa6) spill-free on gfx1100: split the PV accumulator across warps by head dimension, query and K fragments through LDS

## Description

Decode attention at 256-wide heads with six query heads per KV head compiles every instantiation at the 256-VGPR ceiling with spills (VTP37 spill scan; e.g. one-warp form 57 VGPRs / 232 B spilled). Cause by inspection of our own kernel: each warp holds the whole 16 x 256 f32 output accumulator (128 VGPRs), the query fragments (64 VGPRs) and the value prefetch registers, so nothing is left for the chained matrix-multiply operands. Attention decode is 4.5% of the rank 0 traced pass (1.33 ms) and the XTX runs it 3.7-5x slower than the R9700 at the same per-head work; expected lever 0.4-0.8 ms a step on the XTX ranks (VTP37 notes bound it at 1.0-2.2 ms).

**What we will try.** Give each warp a quarter (or half, at two warps) of the output head dimension for the value step, so the f32 output accumulator per warp falls from 128 to 32 (or 64) VGPRs. The score step (query x key) is then the same for every warp of the block, so the key tile is staged once in LDS and read back as matrix fragments with load-count waits that overlap the chained multiplies; the query fragments stay in registers. The softmax running state and the across-block accumulation order are unchanged, so each output column still sums its terms in the same order. Cost to watch: on gfx1100 the repeated score step multiplies the matrix-multiply shim cost, so this is judged only against the accumulate-once baseline (slices attn-decode-acc-once and the native accumulate release).

## Steps

1. Baseline: the release that carries the native-accumulate decode attention.
2. Resource model: for one, two and four warps with the value accumulator split by head dimension, count VGPRs (query fragments, accumulator share, prefetch) and confirm zero spills in a scratch single-unit build for the shipped instantiations.
3. Scratch variant in libr3 native (never a vendor edit); byte identity expected because each column's accumulation order does not change, any difference qualified numerically.
4. Cold timing replay at context 700, 4k and 32k with 1 and 3 query rows, then decode ms/step ABBA at 6 and 8 streams.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

- Resource record: zero spilled VGPRs for the shipped instantiations.
- Output compared with the current kernel on random and recorded inputs.
- Decode ms/step ABBA (untraced, 3 seeds, 6 and 8 streams), acceptance and temp-0 text recorded.

## Effort & Risk

Large. Risk: the repeated score step costs more than the spills it removes on gfx11; the first scratch measurement decides.

## Standards



## Acceptance Criteria

- Ships only if the ABBA gain at 6 or 8 streams exceeds the seed spread or isolated attention time falls at least 25% at context 4k and longer with no short-context regression; otherwise a recorded NO-GO with the measurement.

## Notes

2026-11-24: started with the resource model (step 2) once the accumulate-once baseline is merged and released.

Withdrawn: not justified by our own measurements. Attention decode spills stay tracked under the attn-decode slices and the VTP37 spill scan.

Reinstated 2026-11-24. Ranked against the census table (CENSUS-8.4) after the native-accumulate release lands.

## Change Log

- 2026-11-23T20:42:14.800583+00:00 (created-by): Created by agent
- 2026-11-23T20:43:07.491746+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:effort_risk, section:acceptance_criteria
- 2026-11-23T20:48:32.456232+00:00 (state-transition): State: pending → in_progress
- 2026-11-23T20:54:34.802032+00:00 (updated-by): Updated: section:description, section:steps, section:notes
- 2026-11-23T21:10:52.197017+00:00 (updated-by): Updated: section:notes
- 2026-11-23T21:11:08.007424+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:effort_risk, section:acceptance_criteria
- 2026-11-23T21:11:17.486954+00:00 (state-transition): State: in_progress → deprecated
- 2026-11-23T21:14:27.767439+00:00 (state-transition): State: deprecated → pending
- 2026-11-23T21:15:03.676137+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:effort_risk, section:acceptance_criteria, section:notes
