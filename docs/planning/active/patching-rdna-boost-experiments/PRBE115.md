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

## Change Log

- 2026-09-29T23:14:07.176425+00:00 (created-by): Created by agent
