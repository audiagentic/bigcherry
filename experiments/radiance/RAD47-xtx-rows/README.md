# RAD47: gfx1100 MXFP4 kernel-row selection

Archived from `exp/rad47-xtx-rows` (tip `6de7d8eaf`, base `34550c0b6`), two commits dated 2026-11-04.

## What and why

The gfx1100 XTX selected slow MXFP4 decode, wide GEMV, short-K and skinny `nt_m64` rows for M ≥ 17. The experiment capped decode at 16, made overlapping wide/short-K rows unreachable and ranked skinny `nt_m64` immediately below tiled GEMM.

## Outcome and why stopped

**(a): two changes already on main; no patch.** PR **#596**, main commit `619590728`, contains the same `rewrites.txt` constraints and `native/r4d_rows.cpp.rw` priority rule. The branch is redundant, not rejected. Recorded M=18 tiled ~345 μs versus wide 848 μs, decode 3,581 μs and skinny 1,645 μs; M=64 skinny 3,269 μs versus tiled 530 μs. PR #596 reports +27% 27B throughput at four streams and +37% at six. No hardware was re-measured.

## Evidence and resumption

- `docs/planning/active/patching-radiance-rdna3/RAD47.md` and PR #596.
- Experiment commits `f09597bcb`, `6de7d8eaf`; shipped commit `619590728`.
- Resume from current libr3 row rules; re-run selection tests and qualification only when changing policy.
