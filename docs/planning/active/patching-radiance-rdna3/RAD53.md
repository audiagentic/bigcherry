---
id: RAD53
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-11-14T06:57:48.801584+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: S
---

# gfx1100 fp8 KV read: direct E4M3 to f16 conversion

## Description

gfx1100 has no fp8 conversion instruction. libr3 converts each E4M3 element through `r3_e4m3_to_f32` (`r3_compat.h`, roughly 15 VALU) and a pack to f16, for every K and V element of every attention tile. At 60k tokens on `qwen27b-tp3-int8-556-long` the fp8 cache decodes at 39.4 ms a step against 32.6 ms for bf16 and prefills in 30.7 s against 27.2 s, with half the bytes. E4M3 is exactly representable in f16, so a direct conversion (shift and add on byte pairs for normal codes, a select for exponent-zero codes) is exact.

## Steps

1. `--profile-ops` decode at 60k for fp8 and bf16: attn_paged us/call per rank (prediction: the XTX ranks carry the gap, the R9700 is faster with fp8).
2. A direct conversion in `r3_compat.h` used by `fp8x8_to_16x4w` and the decode kernel's K read; exhaustive 256-code check against `rad_fp8e4m3_to_f32`.
3. kbench decode and prefill attention, fp8 against bf16, one XTX; then TTFT and decode step at 8k / 60k / 100k.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Byte identity of attention output against the current conversion (all 256 codes, selftest rows); greedy identity on the fp8 config; ABBA at depth.

## Effort & Risk

S. Inferred from source and the step delta; not yet profiled per rank (step 1 confirms or kills). Expected fp8 decode step at 60k 39.4 -> 31-33 ms, TTFT at 60k -11%, growing with depth; nothing on the R9700.

## Standards



## Acceptance Criteria

- [ ] fp8 decode step and TTFT at 60k within 3% of bf16 on the same build (or better), or a NO-GO naming the rank and stage that carries the remaining gap.
- [ ] Output bit-identical to the current fp8 path.

## Notes

Created 2026-11-14 (`docs/evidence/2026-11-14-tp3-long-context/README.md`). RAD38 (`k16_v8`) depends on the result: on gfx1100 every 8-bit KV plane pays this conversion. No quality gate: the conversion is exact.

## Change Log

- 2026-11-14T06:57:48.801584+00:00 (created-by): Created by agent
- 2026-11-14T06:58:40.011280+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:effort_risk, section:acceptance_criteria, section:notes
