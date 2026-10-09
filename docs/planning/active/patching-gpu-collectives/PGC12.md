---
id: PGC12
order: 0
plan: patching-gpu-collectives
state: pending
created-at: '2026-10-01T05:46:32.484296+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
---

# Adaptive AllReduce: phase-aware routing (prefill to RCCL, generation to host) and the prompt-tail cost

## Description

Size-only routing in 0840 cannot serve every mode. Plain decode (no MTP, ubatch 512): 1 MiB switch costs pp1024 -4.2% because llama-server splits a 1000-token prompt into 512 + 484 + a 4-token tail whose 128 ARs (80 KB) take the host path; 64 KiB fixes that (pp1024 -0.07%, decode +6.9% kept). MTP depth 5 (ubatch 2048): MTP verify ARs are ~100-120 KB, so 64 KiB moves them to RCCL and loses the whole gain (tg512 +7.9% at 1 MiB vs +0.6% at 64 KiB); at 1 MiB there is no prefill loss. The tail (80 KB) is smaller than verify (120 KB), so no byte threshold separates them. Adaptive is promoted at 1 MiB for the MTP production setup (owner decision 2026-10-01); this item revisits it in depth.

## Steps

1. Explain the tail cost: 128 host calls should cost a few ms, not ~50 ms; pp256 (+0.7%) shows no loss although a tail is expected. Trace per AR {bytes, ne0/ne1, node, ubatch index, provider} for pp256/pp1024/pp4096 (1277_ar_size_trace, extend with node/ubatch), and time the first RCCL->host provider switch.
2. Confirm when llama-server holds back a prompt tail (hybrid-model context checkpoints?) and for which prompt lengths.
3. Design a phase hint from graph execution to the comm context: prefill -> RCCL; generation (decode and MTP verify) -> host below switch_bytes. Do not use ne1==1 as the primary key (prefill also has ne1==1 select tensors). (dev-gpt-agent recommendation.)
4. Implement in 0840 (or a follow-on patch), A/B plain + MTP across pp256/1024/4096, requalify.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

patches/0840_hybrid_allreduce_dispatch/patch.py, patches/1277_ar_size_trace/patch.py, tools/lab/native-vs-patched/server-ab-adaptive-switch*.json

## Validation

Plain and MTP A/B vs RCCL: decode gain kept in both, pp256/pp1024/pp4096 within 0.5%. Decode KLD 0 vs exact f32.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

2026-10-01 profile (prof-27b-q8-2, pp2048 ubatch): per-device compute 0.93 s vs AllReduce 2.4 s per 2048-token ubatch (profiled; ~4 ms/AR unprofiled), start skew between XTX ~0.1 ms, so service not imbalance. Links: XTX PCIe 4.0 x8, R9700 x4, no P2P: host-staged ceiling ~6-7 GB/s, measured 5-10 GB/s. Prefill lever is compute/communication overlap (chunked AR along tokens) or fewer prefill ARs, not RCCL tuning.

## 2026-10-09 PGC13 prerequisite
Before per-topology calibration, prove actual provider completion, phase, graph/ubatch identity, first-provider-switch latency and RCCL linkage. Current lab-only 1277 records `prefer_internal` (predicted route), not successful completion, and its static trace counter is non-atomic. Trace outside timed windows; never infer phase from `ne1`. PGC13 may consume receipts offline, but cannot add a second dispatcher, phase policy or `-ts` tuner. The phase hint, if justified, belongs here and in existing 0840.

## Change Log

- 2026-10-01T05:46:32.484296+00:00 (created-by): Created by agent
- 2026-10-01T07:45:53.626006+00:00 (updated-by): Updated: section:notes
