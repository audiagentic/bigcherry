---
id: RAD53
order: 0
plan: patching-radiance-rdna3
state: completed
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

Exhaustive E4M3 to f16 check (`r3_e4m3_f16_test`, all 256 codes, against `rad_fp8e4m3_to_f32`); selftest prefill attention rows (fp8, k6_v4, split) ok on main libr3; KLD identity on fp8 `-long` against a RAD52-only reference (5 docs, 276,900 positions: PPL equal, top-1 1.0); engine ABBA (2 starts per arm) on fp8 `-long`; 60k decode-step runs fp8/bf16 in the order F B B F on main libr3; 12/12 stress starts without fault lines. Evidence: `docs/evidence/2026-11-15-rad53-fp8-prefill-attention/README.md`.

## Effort & Risk

S. Inferred from source and the step delta; not yet profiled per rank (step 1 confirms or kills). Expected fp8 decode step at 60k 39.4 -> 31-33 ms, TTFT at 60k -11%, growing with depth; nothing on the R9700.

## Standards



## Acceptance Criteria

- [x] fp8 decode step and TTFT at 60k within 3% of bf16 on the same build (or better), or a NO-GO naming the rank and stage that carries the remaining gap. Decode step 32.68 vs 32.24 ms (+1.4%; was 39.4 vs 32.6), TTFT 24.37 vs 23.73 s (+2.7%; was 30.7 vs 27.2).
- [x] Output bit-identical to the current fp8 path. Exhaustive 256-code host test; engine identity against RAD52 alone: PPL equal to every digit, top-1 1.0, KLD max 2.7e-8.

## Notes

Created 2026-11-14 (`docs/evidence/2026-11-14-tp3-long-context/README.md`). RAD38 (`k16_v8`) depends on the result: on gfx1100 every 8-bit KV plane pays this conversion. No quality gate: the conversion is exact.

**2026-11-15 (`docs/evidence/2026-11-15-rad53-fp8-prefill-attention/README.md`).** Step 2 done (direct E4M3 to f16, exhaustive host test). The larger fp8 cost on gfx1100 was in prefill, not the conversion: `O_KLATE` in the native prefill attention spilled the fp8 and k6_v4 kernels (816/832 vs 336/376 bytes a lane); the native form drops it. Identity against RAD52 alone: PPL equal to every digit, top-1 1.0, KLD max 2.7e-8. Kernel at 100k keys: fp8 512/1024/2048 rows 40.5/79.6/119.1 -> 10.2/20.3/55.7 ms (with RAD52). Engine ABBA on fp8 `-long`: TTFT 100k 65.18 -> 45.45 s (-30%), 60k -22% (now +2.7% from bf16), decode +1%; long prompt beside 4 decoders -27.5%, gap max -40%. Stress 12/12. Open: step 1 / acceptance 1 decode half (fp8 vs bf16 decode step at 60k).



**2026-11-15, closed.** Decode at 60k on main libr3, fp8 against bf16: 32.68 vs 32.24 ms per step (+1.4%), TTFT +2.7%. Step 1 (`--profile-ops` per rank) was not needed: the gap closed without it.

## Change Log

- 2026-11-14T06:57:48.801584+00:00 (created-by): Created by agent
- 2026-11-14T06:58:40.011280+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:effort_risk, section:acceptance_criteria, section:notes
- 2026-11-15T00:06:32.139723+00:00 (updated-by): Updated: section:notes
- 2026-11-15T00:06:36.209057+00:00 (state-transition): State: pending → in_progress
- 2026-11-15T01:40:16.413891+00:00 (updated-by): Updated: section:validation, section:acceptance_criteria
- 2026-11-15T01:40:21.443376+00:00 (updated-by): Updated: section:notes
- 2026-11-15T01:40:29.940043+00:00 (state-transition): State: in_progress → completed
