# 1271_prbe54_q5_kv_dequant_f16

**Status:** untested
**Plan item:** PRBE54

## What it does

Adds HIP half2 F16 dequantization for legacy Q5_0/Q5_1 only when FlashAttention stages contiguous KV to F16. The global conversion selector is unchanged.

## Validation

gfx1100/gfx1201; full-vocabulary backend-reference correctness on Q5 KV, subject-only activation, Q5 pp512 prefill positive lane (F16 staging path), Q5 tg128 decode control (direct vector path), 10 paired rounds/session, 4 sessions.

**2026-10-09 gate:** Host Q5_1 double-rounding mismatch 28,261/131,072; Q5_0 0/131,072. No HIP measurement. Q5_0-only producer cannot certify Q5_1. Require Q5_1 baseline parity or disable Q5_1 selector, then prove the actual contiguous F16 staging route and material converter cost. PRBE54 / BCOP79.

## 2026-10-12 historical GPU evidence reconciliation

`evidence/validation.json` already records one b11126 Q5_0 session on each of gfx1201 and gfx1100 (ten paired rounds each); **both contract verdicts INVALID**, not absent. gfx1201 pp512 +0.0143% (CI95 -0.2110..+0.2306), full-vocab pass. gfx1100 pp512 +0.7658% (CI95 -0.0309..+2.0003), full-vocab **FAIL** (material diff 0.342 > 0.02). Neither is a validated gain. The marker is a process-once selector event, not per-lane kernel execution. Producer tests Q5_0 only; Q5_1 double-rounding remains blocked by BCOP79. Do not queue four-session reruns without a current-pin converter critical-path census (>=5% share, theoretical >=3% E2E opportunity) and resolved gfx1100 correctness. Otherwise retire 1271; see PRBE54 / BCOP164. No new GPU test performed.
