---
id: PGC11
order: 0
plan: patching-gpu-collectives
state: pending
created-at: '2026-09-30T14:04:38.045766+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# AllReduce wire precision: f32 vs bf16 vs safe f16, and best-of-all combinations

## Description

Owner request (2026-10-01): investigate the f16 wire option, whether bf16 vs f32 differ significantly, and build combinations that get the best of all options. Context: adaptive with exact f32 host side = +4.1% decode vs RCCL, acceptance equal; bf16 host side ~+6.7% decode, acceptance -2.2 pp; 1272's f16 is a plain cast (overflow -> Inf, unsafe as default); stock RCCL prefill is already bf16 >= 32768 elements; KLD noise floor ~0.

## Steps

1. Tonight's evidence: throughput + acceptance for adaptive x wire (f32/bf16/f16) x 1275 (ab-27b-awl-1/-2); KLD vs host-f32 reference for the same (kld-awl-*).
2. Safe f16: per-block scale or overflow-detect fallback in 1272, with an on-hardware Inf/NaN counter.
3. Best-of-all combos: per-size wire routing (f32 smallest, f16/bf16 mid, RCCL prefill), per-sender format for mixed-arch 3-GPU.
4. Decision rule tying decode gain to KLD/acceptance loss relative to stock RCCL's own bf16 prefill loss; feed the chosen host-side wire into PGC09's promotion.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Balanced A/B (6 rounds) per combo on dual XTX 27B Q8_0 MTP with MTP acceptance per arm; KLD vs host-f32 reference on the frozen 65K-token corpus (noise floor measured each run); safe-f16 must show zero Inf/NaN on the trace counter across the whole KLD corpus.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Design requested from dev-gpt-agent deep-dive session ses_9811a27a734746c3.

2026-10-01 KLD findings (27B Q8_0 dual XTX, 32 x 2048-token prefill chunks, frozen docs corpus, reference = host exact-f32 wire): noise floor (reference repeat) mean 0.000000, p99 3e-5, same-top 99.997% (deterministic). host stock bf16 wire: mean 0.00054, p99 0.0048, same-top 98.98%. RCCL: mean 0.000496, p99 0.0044. adaptive with exact-f32 host side: identical to RCCL (0.000496) — because perplexity passes are all prefill-sized (>= 1 MiB) reductions that adaptive routes to RCCL, and RCCL goes bf16 >= 32768 elements. So prefill-mode KLD measures stock RCCL's own bf16 prefill loss (~0.0005 mean) and CANNOT see the decode-size host wire. Added decode-mode KLD (queue-kld-decode.sh: --ubatch-size 1, 8 chunks, own decode-mode f32 reference) comparing RCCL, host bf16/f16, adaptive f32/bf16. Also measured: RCCL channels (2 / >=8) no gain (8 ch: decode +0.5%, prefill -0.9%); 1275 neutral/-8.5%.

## Change Log

- 2026-09-30T14:04:38.045766+00:00 (created-by): Created by agent
- 2026-09-30T16:25:35.929349+00:00 (updated-by): Updated: section:notes
