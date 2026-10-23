---
id: BCOP181
order: 181
plan: patching-bc-optimizations
state: superseded
created-at: '2026-10-13T08:09:00Z'
created-by: agent
priority: P2
work: S
---

# PGC16: retire duplicate partial-branch AllReduce fold

## Description

## Discovery / disposition

PGC16 proposed `AllReduce(a)+AllReduce(b) -> AllReduce(a+b)` and a new partial-state analyzer/graph rewrite. Pinned llama.cpp `0504396`, b11552 and current upstream already implement that exact two-PARTIAL/ADD delayed-boundary fold inside `ggml_backend_meta_graph_compute::get_i_delayed`; guards cover consumers, shape/type, split state, compute flags, host views and zero-sized shards. The proposed independent pass is **retired**, not implemented. No new inference gain is claimed.

## Owners and residual action

PGC16 is authoritative and now terminal (`done`). PGC15 owns tiled overlap, PGC12 traces, PGC14 transport, PGC09/11 provider/wire, QFP17 prefill evidence. Radiance helper/RAD01, Meta 1340 and ongoing GPU lanes were inside the preceding 12-hour independent-work exclusion and were not modified. Reopen only with a same-pin trace proving a **missed native fold**, a specific safe guard extension and >=3% exposed E2E theoretical ceiling; otherwise no new patch, hardware queue, allocator, registry, scheduler or telemetry.

## Evidence / acceptance

Actual source inspection: 9/9 native fold guards on each of three refs (`0504396`, b11552, master); source-shaped host fixture 12/12 admission/rejection cases, Python compile passed. No repository pytest, HIP build, GPU test or new throughput measurement. Upstream [#30217](https://github.com/ggml-org/llama.cpp/pull/30217) adds host-view handling, not the fold. [SGLang MoE](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/utils.py) supplies negative group/quant/shared-expert gates; vLLM's fused AllReduce+RMSNorm is a different post-collective mechanism. **Terminal acceptance: no duplicate implementation; historical PGC16 proposal superseded.**

## Steps


## Detailed Solution & Technical Design


## Code Samples & Guidance


## Files


## Validation


## Effort & Risk


## Standards


## Acceptance Criteria


## Notes


## Change Log

- 2026-10-23: state 'done' (invalid) corrected to superseded
