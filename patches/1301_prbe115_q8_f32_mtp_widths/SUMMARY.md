# 1301_prbe115_q8_f32_mtp_widths

**Status:** evaluated
**Plan item:** PRBE115

## What it does

PRBE115: widen 1241's Q8_0 raw-F32-activation MMVQ path (no q8_1 activation quantize) from ncols_dst == 1 to the
MTP verify widths, and optionally to RDNA4. Opt-in for screening: `BIGCHERRY_Q8_F32_MAXCOLS=N` (1..8) and
`BIGCHERRY_Q8_F32_RDNA4=1`. The 1241 kernel branch is already column-generic; only the host gate and the
per-width compile-time dispatch change. Activation evidence: 1241's per-ncols `BIGCHERRY_PATCH_HIT` line under
`BIGCHERRY_PATCH_TRACE`.

## Hardware result (2026-10-04 review)

Activation proven; neutral on hardware (QFN01).
