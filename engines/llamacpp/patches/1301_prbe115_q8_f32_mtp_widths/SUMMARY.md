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
