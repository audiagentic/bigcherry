---
id: VTP37
order: 0
plan: run-dense-27b-virtual-tp
state: pending
created-at: '2026-11-12T23:29:57.750388+00:00'
breadth: ''
skill: advanced
created-by: agent
work: L
priority: P0
---

# int8 (w8a8) TP3 decode and prefill: pipelined prefill on rotated-int8 trunks, gfx1100 int8 row selection, three-rank fused AR + norm + int8 quant

## Description

**First priority (owner direction 2026-11-13): raise int8 (w8a8) TP3 decode and prefill.** World: dense Qwen3.8-27B w8a8 (`qwen3.8-27b-w8a8-df2.rad`, plugin `qwen35_w8a8` on the shared `qwen35_fp8` trunk) on XTX + XTX (gfx1100, native int8, emulated fp8) + R9700 (gfx1201), host-staged exact all-reduce, layout `qwen27b-tp3-int8` (kv 1,1,2; ffn 6656/6656/4096; lk 6,6,4; lv 18,18,12; tune_pin all_reduce:nblocks=4,nthreads=256; DFlash2 depth 3; fp8 KV).

**Baseline (2026-11-12, VTP14 N6 refresh, build 8a6f2776, 2x ABBA):**

| | 1 / 4 / 6 / 8 streams tok/s | one-prompt prefill tok/s 512 / 2048 / 8192 |
|---|---|---|
| w8a8 TP2 (two XTX) | 65.3 / 175.9 / 167.9 / 194.4 | 1438 / 1487 / 1903 |
| **w8a8 TP3** | **83.6 / 188.0 / 179.2 / 213.1** | **1547 / 1615 / 2181** |
| MXFP4 TP3 (reference) | 115.6 / 222.7 / 214.2 / 239.3 | 1848 / 1919 / 2709 |

**Why int8 trails MXFP4 TP3 (assumptions to validate, each with its source):**

1. **No pipelined prefill on int8.** `arch/qwen35_fp8/qwen35_fp8.cpp` (~line 1074) leaves `m.pipe_on` false when `m.a_h.q8_fed` (the w8a8 trunk's norms write rotated int8 codes via `rmsnorm_had_quant_i8g`), printing 'the pipelined prefill stays off on this trunk'. So rad0020, rad0025, rad0026 and rad0027 never run on int8. On MXFP4 they were worth: one-prompt TTFT -15 to -19% (rad0020), four-prompt TTFT 10.5 -> 9.0 s and 8 streams 214 -> 240 tok/s (+12%, rad0025+rad0026), a long prompt beside generating streams -19% (rad0027). The stated reason is that `NormQuantFP8::step_plain` (arch/common/rad_fp8.h ~486), used by the pipeline's boundary norms on lane 1, writes `a.q`/`a.s` (E4M3) while `step()` writes `hi8 ? a.q8 : a.q` / `hi8 ? a.s8 : a.s`. `op` is already declared as `rmsnorm_had_quant_i8g` when `hi8`. Hypothesis: the fix is to make `step_plain` write the int8 pair when `hi8`, add `a.q8`/`a.s8` of the pipelined activations to the lane-1 concurrent set (`rad_buf_concurrent`), check every other lane-1 producer (GateQuantFP8 / `gated_had_quant_i8g`, `a_go`, `a_attn`, `a_ffn` q8_fed paths) and drafter taps, then drop the q8_fed refusal.
   - Corollary: VTP31 experiment 12's int8 'pipeline floor 256' gain (+0.3-1.3% decode, -1.7/-2.2% TTFT) cannot be the pipeline (it never ran); treat it as noise and fix the comment on `tuning.pipelined_prefill_min_rows` in the `qwen27b-tp3-int8` layout.
2. **gfx1100 int8 GEMM row selection at M 9-128.** RAD47 fixed this band for MXFP4 on gfx1100 (tiled row 4-17x faster than the skinny/decode rows from M=17; +27% / +37% at 4 / 6 streams on TP2). The int8 rows (`gemm_i8a8_nt_m16`, `gemm_i8a8_tiled`, `gemm_i8a8_rb8_tiled`, `gemm_i8a8_rb32_tiled`) were not swept the same way. RAD48 recorded an open M 33-64 crossover: `gemm_i8a8_nt_m16` 358 us at M=32 and 772 us at M=64 against the native tiled row ~550 us flat (N=34816 K=5120, one XTX). Multi-stream decode with 3 drafts runs M = 4-32 rows per step; the last prefill chunk and mixed steps hit 33-128. Per-rank N is smaller in TP3 (ffn 6656 per XTX rank), so the crossover must be re-measured at TP3 shapes.
3. **No three-rank fused all-reduce + RMSNorm + Hadamard int8 quant.** `NormQuantFP8::declare` declares `ar_rmsnorm_had_quant_i8g` only when `g.world == 2`; TP3 runs `all_reduce` then `rmsnorm_had_quant_i8g` (a separate full-row pass that also does the Hadamard rotation). GPT's estimate for the fp8 twin is under 1% of prefill; the int8 pass is heavier (rotation), so measure its share before building.
4. **int8 kernel efficiency.** RAD48's native gfx11 IU8 tile (on main, #601): one XTX M=2048 N=34816 K=5120 25.8 -> 8.33 ms (3.1x, ~88 TFLOP/s); end-to-end int8 prefill and KLD never recorded. XDP08-A proposes per-row weight-scale LDS staging for `gemm_i8a8_tiled` (only weight scales are unstaged; upstream reports +4% from staging activation scales). Decode GEMV (M 1-8) bandwidth on both archs is unmeasured; int8 weights are ~2x MXFP4 bytes, so 1-stream decode is expected to trail MXFP4 - compute achieved GB/s per rank to see the headroom.
5. **Split.** The int8 split was swept (VTP14 N6); keep it unless the pipeline changes the balance (re-check 6,6,4 vs neighbours after step 1).

## Steps

1. **Validate the assumptions above** from the source and a short profile (`--profile-ops`, int8 TP3, one 2048-token prompt and 8-stream decode): confirm pipe_on is false today, the per-op share of `all_reduce` + `rmsnorm_had_quant_i8g`, the int8 GEMM rows selected at each M per rank, and achieved decode GB/s.
2. **Pipelined prefill for rotated-int8 trunks** as a managed patch (next free number after rad0029, plan-ids VTP37): `step_plain` writes the int8 pair when `hi8`; lane-1 buffer sets include the q8/s8 activations; remove the q8_fed refusal (keep a refusal for any producer that still cannot write int8 on lane 1). Card-free test applying to the pinned source. Build; smoke; greedy identity serial vs pipelined on int8 TP3 (`--pipelined-prefill off` vs `auto`); int8 stress (12 runs, rad0028 guard); 2x ABBA vs the current int8 layout: TTFT 512/2048/8192, four 4096-token prompts, 8192 (or 6144 at max-model-len 8192) beside 4 generating streams, 1/4/6/8 streams.
3. **gfx1100 int8 row crossover:** rad-kbench all-rows on the int8 rows at the TP3 per-rank shapes (N, K from the int8 split) for M 1-256; if a faster valid row loses on priority, change libr3's gfx11 constraints/priority (like RAD47, gfx1201 untouched); ABBA end to end; KLD vs BF16.
4. **Three-rank fused `ar_rmsnorm_had_quant_i8g`** (libr4d `r4d_fused_quant_fp8.hip` / `rad_ar_rmsnorm_quant`, `ar_fused_ok`), only if step 1 shows the separate pass is >= 3% of int8 prefill or decode device time.
5. **int8 tile weight-scale LDS staging** (XDP08-A) and decode GEMV, only if step 1 shows the kernel time dominates after steps 2-3.
6. Fold winners into `qwen27b-tp3-int8`; refresh the baseline with 3x ABBA against int8 TP2 and MXFP4 TP3.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

- Greedy identity: pipelined vs serial on int8 TP3 (same build, same split).
- int8 stress: 12 runs of the rad0028 guard (4/8 streams + 6144-token prompt beside 4 generating streams), zero faults.
- KLD against BF16 for any kernel-row change (rows of a different accumulation order).
- Matched ABBA (2x per lever, 3x for the final refresh): 1/4/6/8 streams with speculation, one-prompt TTFT 512/2048/8192, four 4096-token prompts, long prompt beside 4 streams.

## Effort & Risk



## Standards



## Acceptance Criteria

- int8 TP3 runs the pipelined prefill (startup receipt shows pipe_on) with greedy output identical to serial and no stress faults.
- int8 TP3 one-prompt TTFT improves by at least 10% at 2048 tokens and 4/6/8-stream tok/s by at least 5% over the 2026-11-12 baseline, or each lever's NO-GO is recorded with its measurement.
- Every change is a managed patch or layout setting (no model or card constants), selectable per run.

## Notes

Consolidated 2026-11-13. Carries RAD48's open work (superseded into this item) and XDP08-A (int8 weight-scale staging). Related: VTP36 (TP3 performance generally, MXFP4 2048-token gap), RAD47 (gfx1100 MXFP4 row selection, implemented), VTP32 (split solver). Patched source used for the assumptions: the current build's source tree (rad0016-rad0029 applied). Implementation and validation of the assumptions: handed to a Fable 5.2 agent (2026-11-13).

## Change Log

- 2026-11-12T23:29:57.750388+00:00 (created-by): Created by agent
- 2026-11-12T23:30:35.740511+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:acceptance_criteria, section:notes
