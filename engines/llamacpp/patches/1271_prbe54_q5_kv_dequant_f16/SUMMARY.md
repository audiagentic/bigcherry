# 1271_prbe54_q5_kv_dequant_f16

**Status:** untested
**Plan item:** PRBE54

## What it does

Adds HIP half2 F16 dequantization for legacy Q5_0/Q5_1 only when FlashAttention stages contiguous KV to F16. The global conversion selector is unchanged.

## Validation

gfx1100/gfx1201; full-vocabulary backend-reference correctness on Q5 KV, subject-only activation, Q5 pp512 prefill positive lane (F16 staging path), Q5 tg128 decode control (direct vector path), 10 paired rounds/session, 4 sessions.

**2026-10-09 gate:** Host Q5_1 double-rounding mismatch 28,261/131,072; Q5_0 0/131,072. No HIP measurement. Q5_0-only producer cannot certify Q5_1. Require Q5_1 baseline parity or disable Q5_1 selector, then prove the actual contiguous F16 staging route and material converter cost. PRBE54 / BCOP79.
