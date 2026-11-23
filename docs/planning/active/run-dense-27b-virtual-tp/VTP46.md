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

Source: VTP37 spill scan and a read of a reference gfx1201 decode-attention kernel for the same geometry (24 heads, 4 KV heads, head dim 256, 16-token blocks). Our `r4d_attn_decode_kernel` at h256 gqa6 compiles all 194 instantiations at 256 VGPRs with spills (e.g. NS1 57 / 232 B). Each warp keeps a full 16 x 256 f32 output accumulator (128 VGPRs) plus the query fragments (64 VGPRs) and the prefetch registers. The reference kernel is spill-free at 197 VGPRs: four waves share one K tile staged through LDS, every wave runs the QK chain, the PV step is split by head dimension so each wave holds 4 of 16 output tiles (32 VGPRs), and K fragments are read from LDS with s_wait_dscnt counts that interleave loads with the chained WMMA. Decode attention is 4.5% of the rank 0 traced pass; the XTX is 3.7-5x slower than the R9700 at the same per-head work, about 1.0-2.2 ms a step of lever (VTP37 notes). Caveat: on gfx11 the QK chain repeated in every warp multiplies the WMMA shim cost, so the split only pays together with the accumulate-once change (slice attn-decode-acc-once) and, ideally, native-layout fragments. Not a copy of the reference: it is single-sequence and gathers V with scalar 16-bit loads.

## Steps

1. Land and A/B the attn-decode-acc-once slice first (it is the baseline for this item).
2. Resource model: for NWARPS 2 and 4 with the PV accumulator split by head dimension, compute the VGPR budget (Q fragments, 32-VGPR O tile, prefetch) and confirm zero spills in a scratch single-unit build for the shipped instantiations (fp8 KV, F16, CONTIG, splitkv).
3. Scratch kernel variant in libr3 native (never a vendor edit): PV split by head dimension across warps, S exchanged through LDS or recomputed; byte identity against the current kernel at the same softmax and accumulation order where the order is unchanged, otherwise qualify numerically (the PV order across KV blocks is unchanged, only which warp owns which output columns).
4. Cold timing replay at ctx 700 / 4k / 32k, 1 and 3 query rows, then decode ms/step ABBA at 6 and 8 streams.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

- Resource record: zero spilled VGPRs for the shipped instantiations.
- Output compared with the current kernel on random and recorded inputs (byte identity where the arithmetic order is unchanged).
- Decode ms/step ABBA (untraced, 3 seeds, 6 and 8 streams) with acceptance and temp-0 text recorded; KLD if any order changes.

## Effort & Risk

Large. Risk: the repeated QK per warp costs more than the spills it removes on gfx11; the first scratch measurement decides.

## Standards



## Acceptance Criteria

- The variant ships only if the ABBA gain at 6 or 8 streams exceeds the seed spread (s.e.) or the isolated attention time falls at least 25% at ctx 4k and longer, with no regression at short context; otherwise a recorded NO-GO with the measurement.

## Notes



## Change Log

- 2026-11-23T20:42:14.800583+00:00 (created-by): Created by agent
- 2026-11-23T20:43:07.491746+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:effort_risk, section:acceptance_criteria
