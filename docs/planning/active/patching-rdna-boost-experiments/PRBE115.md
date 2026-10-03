---
id: PRBE115
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-29T23:14:07.176425+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: L
---

# Q8_0 F32-activation MTP-verify kernel for widths 2-5 preserving stock numerics

## Description

GPT ranked candidate #1 (2026-09-30): 1241 proves F32 activation is valuable at width 1, but widening the old RD33 to ne1<=8 changed MTP numerics (acceptance 0.90101->0.95580, greedy divergence), so the widened gate was rejected. Measure stock Q8_1 quantize + MMVQ time by width and its fraction of decode, then design a dedicated verify-width kernel that keeps stock numerical semantics and MTP acceptance. Other ranked candidates to revisit after the profile capture: MMVQ geometry/nwarps for widths 2-5, tensor-split AllReduce latency/payload (after 0860), GDN recurrent/MTP-verify kernel (1254 only helps prefill: K>1, one sequence, n_tokens>K+64), MTP host/sampling/launch overhead.

## Steps



## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Backend-reference per width 2-5, full-vocab/logprob tolerance, exact MTP acceptance parity, 4-session performance on 27B Q8_0 dual XTX.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Gated on the profile-first item; do not start kernel work before the per-width decode fraction is known. Combo acceptance change was explained by widened 1241, not 1206 or 1245.

2026-10-04 result: patch 1301_prbe115_q8_f32_mtp_widths (env-gated BIGCHERRY_Q8_F32_MAXCOLS / _RDNA4) widened 1241's Q8_0 F32-act gate to ncols 1..8 (per-column vec_dot_f32_decode, weights re-dequantized per column). Activation proven (PATCH_HIT ncols=2,3,4) on the Flash-Next deployment candidate. Quick screens (24K): width4 48.8 vs 48.3/49.3 ms/step; +RDNA4 48.2 vs 49.3/48.3 -> neutral; acceptance dropped (171/250, 169/256 vs ~174/242) - the same numerics shift this item already recorded for the rejected widening (repeated without reading this item first). Parked. GPT (RV4215): one bounded try of a hoisted single-dequant variant (gate: kernel -20%, verify4 E2E >= 2%), compared via a replayed draft stream so the verify-width mix is controlled; code requested in req_ee409a9b21e74e86. See patching-qwen-flash-next and QFN01 notes.

## Change Log

- 2026-09-29T23:14:07.176425+00:00 (created-by): Created by agent

## Ledger-events

- chg_20261003_141341_experimental-decode-kernel-pat_2923
- 2026-10-03T14:13:44.858022+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-03T15:21:38.477056+00:00 (updated-by): Updated: section:notes
