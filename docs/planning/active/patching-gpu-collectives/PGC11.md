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

2026-10-01 ab-27b-awl-1 (adaptive 0840 + 1272 wire + 1275 slot_sync=none, 27B Q8_0 dual XTX, MTP n4, 6 balanced rounds). Means f32/bf16/f16: pp1024 995.5/998.8/997.8, pp4096 1248.9/1249.0/1248.8, tg512 92.8/95.5/94.0, tg2048 106.3/104.5/98.6. bf16 vs f32: tg512 +2.86% [+2.54,+3.20], tg2048 -1.65% [-1.94,-1.40]. f16 vs f32: tg512 +1.29%, tg2048 -7.27% [-7.52,-7.02]. MTP acceptance f32 84.28% / bf16 82.17% / f16 78.56% (acceptance gate FAIL for bf16 and f16). Plain f16 wire loses draft acceptance and long-decode throughput: ruled out. bf16 buys short-decode speed with an acceptance cost that turns into a tg2048 loss. f32 stays the adaptive host wire; safe-f16 (scaled) is the only remaining low-precision candidate. awl-2 row failed only because the running queue script predates the consolidation (stale reference, no rerun needed).

2026-10-02 recorded (run finished overnight, log ~/bc-runs/queue-kld-decode-0225.log): decode-mode KLD (ubatch 1, 8 chunks, own decode-mode host-f32 reference), 27B Q8_0 dual XTX. ref repeat: mean 0.000000, p99 0.00003, same-top 99.99%. RCCL: 0.000000 / 0.00003 / 99.99% (decode-size reductions are below RCCL's bf16 threshold, so f32). host bf16: 0.000020 / 0.00017 / 99.74%. host f16: 0.000003 / 0.00008 / 99.94%. adaptive f32: 0.000000 / 0.00003 / 99.99%. adaptive bf16: 0.000020 / 0.00017 / 99.74%. All PASS the KLD gate, but the MTP acceptance A/B (ab-27b-awl-1) already fails bf16 (82.17% vs 84.28%) and f16 (78.56%), so acceptance, not KLD, is the binding gate. Conclusion unchanged: f32 stays the adaptive host wire; safe-f16 (scaled) is the only remaining low-precision candidate (unbuilt). Note f16 has lower KLD than bf16 yet the worst acceptance and tg2048 (-7.27%) - KLD at decode size does not predict MTP acceptance here.

2026-10-04 consolidation: owner item for all AllReduce wire-format work. Merged here (now superseded): PGC07 (bf16/fp8 wire on --allreduce-wire), PNRO01 (Q8_0 wire for the internal HIP AllReduce; patch 1250 is a scaffold only and currently FAILED_NEEDS_RECONCILIATION), RNX07 (WHT + low-bit compression over existing providers, CPU-root/SHM on no-P2P). Keep their constraints: correctness/work-equivalence evidence before speed; do not port r9700 P2P/IPC transport. Flash-Next context (QFP01, RV4214): decode ARs are 10-40 KB and already on CPU-root (1291); the wire-format lever is large prefill ARs (~96 x 10.5 MB per 1024-token ubatch, ~30-40% of prefill). Online: TensorRT-LLM AllReduceStrategy.LOWPRECISION for PCIe without NVLink; block-FP8/int8 with Scale+Hadamard ~3.5-4.5x byte reduction; fuse pack into the producer epilogue. GPT estimate +7-14% prefill. Gate: a standalone 3-rank F32->BF16->RCCL->F32 3.3 MB replay must beat ~0.9 ms vs 1.34 ms before model integration.

## Change Log

- 2026-09-30T14:04:38.045766+00:00 (created-by): Created by agent
- 2026-09-30T16:25:35.929349+00:00 (updated-by): Updated: section:notes
- 2026-09-30T17:27:22.793498+00:00 (updated-by): Updated: section:notes
- 2026-10-02T05:34:08.182339+00:00 (updated-by): Updated: section:notes
- 2026-10-03T15:21:42.101340+00:00 (updated-by): Updated: section:notes
