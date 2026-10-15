# 1301_prbe115_q8_f32_mtp_widths

**Status:** rejected
**Plan item:** PRBE115

## What it does

PRBE115: widen 1241's Q8_0 raw-F32-activation MMVQ path (no q8_1 activation quantize) from ncols_dst == 1 to the
MTP verify widths, and optionally to RDNA4. Opt-in for screening: `BIGCHERRY_Q8_F32_MAXCOLS=N` (1..8) and
`BIGCHERRY_Q8_F32_RDNA4=1`. The 1241 kernel branch is already column-generic; only the host gate and the
per-width compile-time dispatch change. Activation evidence: 1241's per-ncols `BIGCHERRY_PATCH_HIT` line under
`BIGCHERRY_PATCH_TRACE`.

## Hardware result (2026-10-04 review)

Activation proven; neutral on hardware (QFN01).

## Re-test on the right model and rejection (QFP21, 2026-10-05)

Re-tested on Qwen3.8-27B Q8_0 dual-XTX tensor split, built-in MTP4 (where Q8_0 dominates decode), promoted base,
ABBA at 10K and 32K, per-width activation confirmed (`ncols=1..5` hits): MAXCOLS 2 neutral (73.1 vs 73.0/73.2 t/s at
10K, 72.0 vs 72.1/72.3 at 32K), MAXCOLS 3 neutral (72.7-73.2 vs 72.8-72.9; 72.0-72.1 vs 72.0-72.3), MAXCOLS 5 regressed
(64.7-64.8 vs 73.0-73.4 at 10K, -12%; 67.8-67.9 vs 72.0-72.1 at 32K, -6%). Raw-F32 Q8_0 MMVQ at MTP verify widths is
never faster than quantize-to-Q8_1 + MMVQ here; no width helps. Rejected (runs: qfp21-1301-27b, -w2b, -w3b).

## 2026-10-14 BCOP200 terminal correction (source/host audit only)

Do not revive the rejected generic raw-F32 width2–5 gate or the old "hoisted single-dequant" proposal. Stock b11552 `ggml_cuda_mul_mat_vec_q` quantizes Q8_1 once and shares it across weight rows; 1301 still executes raw-F32 FMA and therefore cannot guarantee stock MTP-verify numerics. The 1241 width1 path remains separately validated and depends on 0600 MMVQ geometry. PRBE115 now requires PRBE113's physical-rank-bound stock width census and >=2.9126% *exposed removable latency* before any default-off exact-Q8_1 producer candidate. Reuse existing ownership; no new kernel, queue or measured speedup. Source assertions 12/12; illustrative host checks 6/6. No patch application, HIP build or GPU run.
