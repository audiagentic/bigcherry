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

## Change Log

- 2026-10-01T05:46:32.484296+00:00 (created-by): Created by agent
