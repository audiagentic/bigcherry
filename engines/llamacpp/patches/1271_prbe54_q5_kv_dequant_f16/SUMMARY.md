# 1271_prbe54_q5_kv_dequant_f16

**Status:** untested
**Plan item:** PRBE54

## What it does

Adds HIP half2 F16 dequantization for legacy Q5_0/Q5_1 only when FlashAttention stages contiguous KV to F16. The global conversion selector is unchanged.

## Validation

gfx1100/gfx1201; full-vocabulary backend-reference correctness on Q5 KV, subject-only activation, Q5 pp512 prefill positive lane (F16 staging path), Q5 tg128 decode control (direct vector path), 10 paired rounds/session, 4 sessions.

**2026-10-09 gate:** Host Q5_1 double-rounding mismatch 28,261/131,072; Q5_0 0/131,072. No new HIP measurement in that audit; archived b11126 GPU receipts are reconciled below. Q5_0-only producer cannot certify Q5_1. Require Q5_1 baseline parity or disable Q5_1 selector, then prove the actual contiguous F16 staging route and material converter cost. PRBE54 / BCOP79.

## Archived GPU evidence (2026-10-14 reconciliation)

`evidence/validation.json` contains one b11126 Q5_0 session per architecture: gfx1201 pp512 +0.0143% (CI95 [-0.2110,+0.2306]), full-vocab PASS; gfx1100 +0.7658% (CI95 [-0.0309,+2.0003]), **full-vocab FAIL** (material diff 0.342 > 0.02). Both fail the required four-session contract; no validated gain. The gfx1100 cause is unknown. Q5_1 half2 rounding remains host-ineligible. `untested` means not validated, not never exercised. PRBE54 now requires gfx1100 A/A vs A/B correctness and exposed converter-cost admission before further hardware sessions. No new HIP run in this audit.
