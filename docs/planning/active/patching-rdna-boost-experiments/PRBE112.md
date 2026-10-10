---
id: PRBE112
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-25T23:16:25.986252+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: M
---

# RD39–42: gate rejected shared-expert stream overlap against native fused MMVQ

## Current disposition and evidence (2026-10-10)

1215_rd394041_amd_stream_moe_overlap is **rejected**, not a production candidate. The later b11126 four-session campaign reported target tg128 **−2.8..−3.3% gfx1100 / −8.0..−8.2% gfx1201** and dense control **−1.6..−2.2% / −6.8%**; `1215/evidence/validation.json` records failed contracts despite activation and full-vocabulary parity. The early +2.38% single-card 10-round README result used a different historical qualification and cannot override the rejection. Do not queue 1215 or reuse its earlier PASS.

## New implementation-level finding

Pinned llama.cpp b11474 (`0504396140d1`), `ggml/src/ggml-cuda/ggml-cuda.cu`:
- `ggml_cuda_match_shared_expert` (~1826–1873) recognizes six routed/shared GEMM+GLU nodes with strict quantization, shape, contiguous layout and one-row conditions.
- `ggml_cuda_try_fuse` (~3502–3528) dispatches **one fused Q8 MMVQ** producing routed and shared outputs through `ggml_cuda_mul_mat_vec_q(...,&fusion)` only when `cuda_ctx->stream_context().concurrent_events.empty()` and fusion-memory checks pass.
- `ggml_backend_cuda_graph_optimize` (~4712–4953) creates stock QKV concurrent events for `attn_norm` fan-out 3, but only with `GGML_CUDA_GRAPH_OPT=1` and enabled CUDA graphs.
- Rejected 1215 `patch.py::_MOE_OVERLAP_NEW` creates an RD42 shared-expert concurrent event via `concurrent_events.emplace(fork_node,...)`, maps shared work to an auxiliary stream and redirects selected tensors to `concurrent_scratch`.
- `ggml_cuda_mul_mat_cublas_impl` already uses `ctx.stream()` and `ctx.cublas_handle()` (~1408–1422); upstream PR #26574 supplies per-stream workspaces. Do not re-port RD39/RD40.

**Proven:** a nonempty event map blocks the native fused routed/shared path. **Not proven:** that a real BigCherry graph satisfies the native matcher, that RD42 disables its actual launches, or that this explains the old regression. Dense-control regression prevents attributing the loss solely to MoE overlap. Treat native fusion and dual-stream overlap as competing mechanisms until profiled.

## Cheapest discriminator (no implementation yet)

1. **Completed:** 18/18 source-static/host-admission checks for native `concurrent_events.empty()`, `&fusion` dispatch, graph-opt gates, 1215 event insertion, active-stream implementation and patch metadata. Host truth table: eligible+zero events+safe memory => fused; any QKV/RD42 event, incompatible shape or unsafe memory => no fusion. This is not a GPU measurement.
2. **First hardware gate, only when existing host-exclusive queue is idle:** reuse 0700/0810 diagnostics or disposable logs, not a new registry. Record graph UID, eval/phase, n_tokens, GPU, graph-opt and CUDA-graph state, QKV/RD42 event counts, native matcher candidates, rejection reasons (events/shape/memory), **completed** `&fusion` launches, kernel counts/durations, overlap intersection, graph rebuilds, scratch and peak VRAM. A marker alone does not prove completed work.
3. **Two stock controls first:** A=b11474 production graph-opt OFF; B=same binary graph-opt ON. Same Qwen3.6-35B-A3B Q4_K_M single gfx1100 XTX, model hash, context, quant, ubatch and clocks; include a dense control. If graph-opt causes >1% dense regression, native fusion is lost without net gain, or no eligible topology exists, stop and close.
4. Only if A/B survives: C=stock+standalone 1216 graph-opt ON (PRBE35 correctness owner); D=1215+1216 **disposable diagnostic only** if b11474 composition is proven. Never promote D. Fail closed on missing activation, changed work count, failed full-vocab parity or graph lifetime.
5. Bound optimistic E2E gain by measured shared-branch critical-path time minus launch/event overhead and any native fusion opportunity lost. If <3%, **close PRBE112**.

## Conditional implementation design

Only for a proven native-fusion-ineligible single-card decode graph, consider a **new** opt-in RD42-only patch using stock concurrency. Do not copy 1215's RD39/40/41 scaffolding.

```text
if !single_gpu || tokens != 1 || !cuda_graph_enabled: fallback
if native_shared_fusion_eligible || conflicting_event_region: fallback
match typed routed/shared ADD join, common fork, exact consumers
prove independent writes, allocator ownership and graph-generation lifetime
fork shared branch to one aux stream; routed branch stays stream0
record producer readiness; join device event before ADD consumer
preserve independent PRBE35 join-fusion guard; fail closed if unavailable
on replay: verify event-slot generation and live buffer ownership
```

CUDA backend context owns streams/events; ggml scheduler/allocator owns tensors. No speculative `ggml_tensor::data/buffer` redirection, new scheduler, allocator, dispatch table, telemetry registry, cache or configuration surface. Single gfx1100 only; no P2P/RCCL/tensor-split inference. gfx1201/gfx1030 need independent qualification.

## Correctness/performance gates

Pristine b11474 patch composition, negative eligibility fixtures (no shared expert, native fusion eligible, >1 token, graph disabled, unsupported layout, multi-GPU), exact full-vocab logits where possible, otherwise approved KLD/logprob/greedy thresholds, MTP acceptance, repeated same-process requests, multi-ubatch, long-context rollover, graph capture/replay, scratch red zones and expected-versus-observed work. First prove critical-path contribution. Only then four independent sessions and >=10 paired ABBA rounds/session. Promotion requires **CI95-low >=3% end-to-end decode over best eligible native control**, dense regression <=1%, no material prefill regression and preserved correctness. Otherwise retire without new implementation.

## Ownership and upstream/forks

PRBE35/1216 owns stock QKV join safety: current 1216 `patch.toml` has `requires=[]` and a contract, but stored b11126 PASS used rejected 1215 in its control; it is **not standalone-qualified**. QFP35 owns native fusion; PRBE113 owns profiling. Do not modify their protected implementation/queue.

Upstream llama.cpp #16991 (QKV concurrency, merged 2025-11-30), #26574 (per-stream cuBLAS workspace, merged 2026-08-20), b11474 source; AMD-Ecosystem fork #36 historical. SGLang `qwen2_moe.py::forward_normal_dual_stream` explicitly joins alt/main streams; vLLM `moe_runner.py::_apply_quant_method` launches `maybe_forward_async` then `wait`. vLLM ROCm #38665 is open; #48111 warns HIP graph capture needs more than `is_cuda_alike()`. NVIDIA/gfx950 gains are not RDNA3 evidence.

## Change Log

- 2026-10-10 (BCOP103): native-fusion exclusion and bounded terminal gate; 18/18 host/static checks. No current-pin build/GPU run.
- 2026-10-08: triage pending, 1215 rejected.
- 2026-09-25: created.
