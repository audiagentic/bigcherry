---
id: PGC12
order: 0
plan: patching-gpu-collectives
state: deprecated
created-at: '2026-10-01T05:46:32.484296+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
---

# Adaptive AllReduce: phase-aware routing (prefill to RCCL, generation to host) and the prompt-tail cost


## Authoritative disposition — 2026-10-15, b11552 (BCOP238)

**DONE: reject an unproven phase-aware provider policy.** The historical Description/Steps below refer to mixed 1 MiB, 64 KiB, BF16 and exact-F32 arms; they do not establish that two phases on the *same topology, exact wire, pinned binary and message size* require opposite providers. Keep the validated dual-gfx1100 exact-F32 96 KiB scalar crossover. Do not add a phase flag, phase-specific dispatcher, threshold table, per-request cache or graph mutation. The remaining provider-outcome receipt belongs to PGC09/0830 with optional lab-only 1277; offline scalar comparison remains PGC13. This item is terminal as a phase-policy design, not a claimed performance gain.

### Current source truth, ownership and lifetime

1. engines/llamacpp/patches/0860_allreduce_provider_cli/patch.py sets ADAPTIVE_SWITCH_BYTES_DEFAULT = 96 << 10 and applies configuration after argument parsing. engines/llamacpp/patches/0840_hybrid_allreduce_dispatch/patch.py snapshots the threshold at ggml_backend_cuda_comm_init_hybrid into comm_ctx->adaptive_switch_bytes; it forces the internal path's BF16 threshold to zero, making the host reduction exact F32. This is a per-communication-context immutable input, not a phase hint.
2. ggml_backend_cuda_comm_try_allreduce_hybrid computes reduction_bytes = ggml_nbytes(tensors[0]). If internal is available and (bytes < switch_bytes or RCCL is absent), it tries internal. A declined internal preflight can fall through to RCCL. If RCCL is selected, its bool return is propagated; there is no post-RCCL host replay. The existing internal ggml_backend_cuda_comm_allreduce_internal checks type, rank shape, contiguity and alignment before calling ggml_cuda_ar_allreduce; the pinned copy path uses host_large/dev_tmp, events and stream ordering. A hypothetical provider change must preserve these guards and may never replay after an ambiguous submitted collective.
3. ggml/src/ggml-cuda/ggml-cuda.cu::ggml_backend_cuda_comm_allreduce_tensor is the provider boundary. Optional 0830_split_reduce_telemetry wraps actual provider return, but cannot infer the llama request phase/ubatch/graph UID from the reduction tensor. The 1277_ar_size_trace insertion occurs **before** dispatch, logs prefer_internal rather than provider completion, and uses a non-atomic static trace_count. It is a lab hint only, not an execution witness or concurrency-safe trace. Neither provider_name set before the call nor a startup BIGCHERRY_PATCH_HIT proves completed AllReduce.
4. PGC09/0840/0860 owns runtime provider/threshold, PGC09/0830 with 1277 owns any bounded provider-completion receipt, PGC13 owns offline scalar comparison, PGC11 owns wire/acceptance, PGC14 RCCL transport, PGC15 tiling, and QFP01/1291 N=3 CPU-root. No second scheduler, event ring, dispatcher, allocator, cache or telemetry schema is authorised. gfx1030 auxiliary and gfx1201 R9700 are not in the auto-admitted dual-gfx1100 envelope; no normal P2P exists on the dev mixed topology.

### Source-derived route matrix (not a benchmark)

For representative F32 messages on dual gfx1100 with RCCL linked and both providers available:

| Workload example | Logical reduction bytes | 64 KiB threshold | **96 KiB production** | 1 MiB threshold |
| --- | ---: | --- | --- | --- |
| Plain decode | 20 KiB | internal | internal | internal |
| 4-token prompt tail | 80 KiB | RCCL | internal | internal |
| 6-token MTP verify | 120 KiB | RCCL | RCCL | internal |
| Large prefill | 10 MiB | RCCL | RCCL | RCCL |

The 96 KiB threshold already separates plain decode and verify sizes. A monotone host-below-threshold rule cannot send the *smaller* 80 KiB tail to RCCL while sending the *larger* 120 KiB verify to internal; that mathematical observation does **not** establish that such a nonmonotone policy is desirable. Actual shapes and phase must be witnessed, not assumed from ne1 or these examples.

### Existing first-party measurements and falsification of the original premise

- Historical plain, no-MTP pp1024 at 1 MiB: about **-4.2%** versus RCCL; 64 KiB: about **-0.07%**. The 4-token tail's 128 small reductions were routed to internal at 1 MiB. The ~50 ms mechanism was **not** isolated; do not call it measured provider-switch overhead.
- Historical 1 MiB MTP5 **BF16** host-wire tg512: +7.98%, with acceptance 78.45% versus 78.74%; this is **not** the production exact-F32 wire. Later matched exact-F32 MTP runs found 1 MiB about **-1.4%** and 96 KiB about **+0.3%**. Thus the earlier supposed need to route MTP verify to host is not demonstrated under the actual production precision.
- Validated t-0840e-gfx1100-s1..s4, exact-F32 96 KiB: **tg128 +3.87%..+3.98%**, **pp512 +0.02%..+0.04%**, bit-identical. These are archived first-party measurements, not fresh results or a phase-policy win. N=3 CPU-root, gfx1201 and BF16-wire observations are not interchangeable controls.

### Cheapest discriminator and conditional reopening (owned by PGC09/PGC13)

1. First reuse completed production receipts. Only if a reproducible bottleneck remains, extend the existing 0830 outcome seam/1277 lab trace with **request/chunk/graph UID, authoritative prefill-vs-decode-vs-verify phase, physical rank UUID/order, type/wire/bytes, requested/preferred/attempted/completed provider, RCCL linkage, first-provider transition, host wait and event generation**. Join to existing 1319/1320/1325 and rocprof records outside timed runs. Trace IDs must be per-request/per-graph; avoid static unsynchronised counters, synthetic provider success and cross-rank summed GPU time. Distinguish preflight decline from post-submit failure; fail closed on ambiguous side effects.
2. On one identical binary/GGUF/topology, prove **two phases at overlapping message size and exact wire** have opposite measured optimal providers with at least **3% exposed end-to-end** removable latency (required serial fraction >= 1-1/1.03 = 2.9126%). Otherwise terminal NO_PHASE_CONFLICT / BELOW_CEILING. If this proof fails, PGC13 may independently close its scalar-calibration proposal; neither owner queues new hardware work.
3. Only on positive proof, PGC09 may prototype an **opt-in, per-call phase tag** supplied by llama_context::process_ubatch or its existing graph submission owner, carried with generation/graph identity into the same communication context. Unknown, mixed-phase, reused/stale graph, cancelled or split/reshaped inputs must use existing 96 KiB size routing. No global mutable phase, ne1 heuristic, graph-capture-unsafe host callback or second dispatch table. Preserve inactive-shard zeroing, RCCL F32/BF16 codec threshold, all-rank ordinal and event/slot lifetime.
4. Gate on pp256/512/1024/4096, tg128/512, MTP verify 2–6, 8K/98K, two sequential same-process requests, two slots, ubatch/context shift, graph capture/replay, cancellation, logits/KLD/greedy/acceptance, physical traffic and no skipped reductions. Compare against exact-F32 96 KiB, 64 KiB and 128 KiB scalar controls; >=4 independent sessions and >=10 paired ABBA rounds, CI95-low >=3% E2E on the bottleneck, <=0.5% prefill and <=1% decode regression. Failure => REJECT_PHASE_POLICY; no default change without this gate.

### Upstream and fork decision (source/diff checked 2026-10-15)

- llama.cpp HIP internal AllReduce #27825 is already the pinned host-staged baseline; #29793 supplies inactive-shard FILL correctness. Open #28967 adds per-process/multi-node communication API, not a phase-aware crossover; do not port it for this two-GPU question.
- vLLM #61848 (opened 2026-10-15) adds page-aligned uncached IPC, release/acquire peer visibility and RDNA4 P2P size gates. Its reported 500/500 stale-read reproductions on **P2P-capable** R9700 systems are a negative graph-replay correctness reference, not transferable speedup or a reason to enable P2P on dev.
- vLLM #57042 changes two-stage rank accumulation order to avoid size-dependent bit drift; its adversarial order-sensitive tests are a useful numerical gate, not a drop-in HIP provider. SGLang PCIe-IPC uses shape/workspace admission on NVIDIA PCIe systems and is not eligible on this no-P2P topology.
- Sources: https://github.com/ggml-org/llama.cpp/pull/27825 ; https://github.com/ggml-org/llama.cpp/pull/29793 ; https://github.com/ggml-org/llama.cpp/pull/28967 ; https://github.com/vllm-project/vllm/pull/61848 ; https://github.com/vllm-project/vllm/pull/57042 ; https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/distributed/device_communicators/pcie_ipc_ar.py

**Validation this audit:** 16/16 source assertions and 14/14 disposable host-only route/admission tests passed; Python compilation passed. No repository pytest suite, HIP build, GPU run, new profiling or benchmark. Historical evidence only. The original plan below is superseded.

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
- 2026-10-23: state 'done' (invalid) corrected to deprecated
