---
id: PGC08
order: 0
plan: patching-gpu-collectives
state: pending
created-at: '2026-09-29T09:34:11.636242+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# 1254 GDN MTP prefix tail: validate 1253 vs 1253+1254 (and 1261)

## Description

CORRECTED: 1254_nro05 patch.py is already fully implemented (eligibility guard, opt-out GGML_CUDA_GDN_CHUNKED, RDNA3/4 bf16 prefix, sequential K-token tail, marker patch=1254_nro05 path=gdn_mtp_prefix_bf16); its README is stale (as is 1253's). No routing to author. Work: fix both READMEs; build 1253-only control and 1253+1254 subject; paired order-balanced A/B on the production shape (Qwen3.8-27B Q8_0, dual gfx1100, -sm tensor, MTP n_max=4), marker required on subject and absent on control, acceptance parity, greedy token parity. Expect little steady-state tg gain: 1254 mainly speeds prompt-side MTP prefill. Also 1261: add ctx_other_backend trace marker and validate control vs +1261 (may be a no-op on a symmetric split).

## Steps



## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

ab-benchmark dual gfx1100, -sm tensor, MTP n_max=4; token parity; marker per arm.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes



## Change Log

- 2026-09-29T09:34:11.636242+00:00 (created-by): Created by agent
- 2026-09-29T09:49:36.115437+00:00 (updated-by): Updated: section:title, section:description

## Ledger-events


- chg_20260929_135722_allreduce-methods-are-now-sele_7144
- 2026-09-29T13:57:38.087570+00:00 (updated-by): Updated: section:ledger-events
- chg_20260929_215656_dual-xtx-27b-q8_0-plain-decode_1707
- 2026-09-29T21:57:14.630409+00:00 (updated-by): Updated: section:ledger-events
