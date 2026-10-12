---
id: PRBE67
order: 0
plan: patching-rdna-boost-experiments
state: done
created-at: '2026-09-09T10:58:14.009489+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: P2
---

# HIP-GRAPH-002: close speculative FA UID-recapture patch; preserve evidence-triggered re-entry

## Disposition (2026-10-12 UTC)

**Close PRBE67 without a new fingerprint, recapture patch, graph bypass, hardware queue, or runtime flag.** A graph with a nonzero UID and changed node/source properties would evade the HIP/CUDA graph property check, but no BigCherry long-context failure has been shown to have that identity transition. The ordinary scheduler *assigns new UIDs when it splits/allocates graphs*; the Meta backend assigns new UIDs when it rebuilds per-device subgraphs. Therefore the original claim that first-node-pointer cache keys alone permit arbitrary stale replay is not established for supported scheduler lifetimes. This is an evidence-based no-change disposition, **not proof that HIP graphs can never fail**. Preserve the bounded re-entry gate below; PRBE66 is already completed/merged into this item.

## Implementation and ownership audit

- Pinned `config/recipes.toml` uses **b11474**. `ggml/src/ggml-cuda/ggml-cuda.cu::ggml_cuda_graph_get_key` indexes the cache with `cgraph->nodes[0]`; it does **not** itself authorize replay. `ggml_cuda_graph_update_required` first returns false on matching nonzero `cgraph->uid == graph->uid` (after asserting equal node count), otherwise compares node count, a copied `ggml_tensor`, and every source's data pointer, `ne[]` and `nb[]`. A changed property with **the same nonzero UID** is the sole demonstrated source-level bypass of that comparison. The function body is identical at b11474, b11552 and current upstream master as inspected 2026-10-12; no new upstream fix was found for this predicate.
- `ggml/src/ggml-backend.cpp::ggml_backend_sched_split_graph` sets `graph->uid = ggml_graph_next_uid()`; it also assigns fresh UIDs to every split graph. `ggml_backend_sched_alloc_graph` calls `split_graph` before dispatch. `ggml_graph_next_uid` in `ggml/src/ggml.c` is an atomic monotonic counter. In `ggml/src/ggml-backend-meta.cpp::ggml_backend_meta_graph_compute`, a changed parent UID triggers subgraph reconstruction and `cgraph_ij->uid = ggml_graph_next_uid()`; unchanged UID explicitly means reuse of the existing subgraph. UID zero takes the CUDA full-property-comparison path. **Inference:** supported rebuilds normally invalidate UID equality. It remains possible for an independent producer to mutate a live graph without updating its UID; no such producer was identified or reproduced for PRBE67.
- Existing graph warmup/capture path in `ggml_backend_cuda_graph_compute` already resets warmup when the full property comparison detects changes; `ggml_cuda_graph_update_executable` re-instantiates on `cudaErrorGraphExecUpdateFailure`. A speculative FA-only hash of `ne[]` is weaker than the existing comparison and would add another cache/identity contract.
- `1233_rd73_stable_graph_cache_key` already tried a shape-derived cache key and was **rejected** after a better-controlled balanced hardware A/B: dual-gfx1100 Qwen MTP tg128 **102.02 -> 100.02 t/s (-1.96%)**, tg512 **100.34 -> 99.09 (-1.24%)**, pp1024 **987.36 -> 980.52 (-0.69%)**. Earlier +1.889% governed evidence remains in that patch's SUMMARY but is superseded by HI162's later controlled regression. These numbers are **not PRBE67 measurements**.
- `1302_cuda_graph_oom_evict` (QFN03, validated) handles graph-instantiation OOM by evicting cached executables; `1304_cuda_graph_lru_cap` (QFP06, rejected) caused recapture churn below the measured working set. Neither fixes or demonstrates UID staleness. PRBE66's selective graphs-off fallback is completed and consolidated here; no second graph owner, scheduler, allocator, key registry, or telemetry system is authorized.
- Historical gfx1201/rocWMMA failure in upstream issue #24961 was on an obsolete b9553 legacy kernel and does not establish this UID cause. Issue #23579 describes a different multi-device capture restriction. Draft upstream PR #29287 implements **RPC serialized-graph caching by UID**, not a local HIP graph-recature fix; it is unmerged and its external APU/network speed claims do not transfer to no-P2P dual-XTX/R9700. vLLM's fixed capture sizes and persistent input buffers are a useful lifetime/identity precedent, not a drop-in patch.

## Cheap discriminator completed

Source-static checks against b11474/b11552: **13/13 passed** (UID fast path, full tensor/source comparison, scheduler/Meta UID creation and atomic generator). Disposable host control-flow model: **8/8 cases passed**, including zero UID, fresh UID, unchanged UID and deliberately mutated same-UID properties. The latter demonstrates a *conditional* bypass, not an observed invalid graph replay. No BigCherry repository pytest, C++/HIP build, GPU run or benchmark was executed in this audit.

## Conditional re-entry — not queued

Reopen only if a **new, current-pin** gfx1201/R9700 long-context FA/graph failure is reproduced on the same workload with graphs ON and absent with graphs OFF, **and** correlated trace evidence shows a reused nonzero UID with changed node/source properties before a stale launch. Instrument only the existing `ggml_cuda_graph_update_required` / `ggml_backend_cuda_graph_compute` path and existing profiling/graph receipts: request/ubatch identity, device, graph key, old/new UID, node count, FA K `src[1]->ne[1]`, K data pointer and strides, property-compare branch, warmup/reset, capture/update result, recapture count, graph-instance lifetime, GPU error and peak VRAM. Capture the same evidence for stable-shape controls. A graph-on/off difference alone is **not** sufficient to attribute UID staleness.

If the trace shows **no same-UID property change**, close this hypothesis permanently and triage the actual failure with its existing owner (capture restrictions, FA kernel, MTP sequencing or 1302 OOM). If it **does** show a change, first create a deterministic host/backend regression with a stable first-node key, changed FA source and unchanged UID. Then prototype the **minimum guard** that falls through to the existing full node/source comparison on a changed signature; avoid a second fingerprint map and do not weaken non-FA graph identity. Prefer repairing the producer's UID lifetime if that is the actual contract violation. Use graphs-off only as a control/rollback.

Before promotion: repeated same-process requests and multi-ubatch, 8K/80K/>=160K context, gfx1201 subject plus gfx1100 control, F16/quant KV, MTP acceptance when applicable, full-vocabulary logits/greedy parity against graphs-off, zero stale replay, expected-vs-observed graph work, bounded recaptures and memory, and <=1% stable-control regression. A demonstrated correctness repair may be adopted without speed gain; a performance-only change requires CI95-low >=3% end-to-end across independent paired sessions. Otherwise reject and leave upstream behavior unchanged. Do not overlap current Radiance attention/GDN, AllReduce, MoE, n-gram, prefill, QFP15 or PA44/PA49 work.

## References / provenance

- llama.cpp b11474/b11552: `ggml/src/ggml-cuda/ggml-cuda.cu`, `ggml/src/ggml-backend.cpp`, `ggml/src/ggml-backend-meta.cpp`, `ggml/src/ggml.c`.
- https://github.com/ggml-org/llama.cpp/issues/24961 ; https://github.com/ggml-org/llama.cpp/issues/23579 ; https://github.com/ggml-org/llama.cpp/pull/29287 ; https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/llm_base_proposer.py
- PRBE66 (completed), PRBE67 history (2026-09-09 through 2026-10-07), QFN03/1302, QFP06/1304, RD73/1233, HI162, BCOP166. Earlier contradictory TODO/fingerprint steps remain in Git history, not as live instructions.

## Change Log

- 2026-10-12: Closed speculative UID-recapture work after tracing scheduler and Meta UID ownership; conditional re-entry only on new correlated evidence. No implementation or new benchmark claimed.
- 2026-10-08: PRBE backlog triage left this pending with a UID-fast-path hypothesis.
- 2026-10-07: Previous audit identified the same-UID early-return path but did not trace ordinary scheduler UID renewal.
- 2026-09-24: Prior review disproved the blanket no-topology-comparison premise.

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.430745+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.261223+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:17:52.882598+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_031805_repaired-four-more-graph-and-v_2834
- 2026-09-10T03:18:05.362966+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:33:42.641725+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes
- 2026-09-24T04:47:06.065592+00:00 (updated-by): Updated: section:description, section:steps
- 2026-09-24T04:47:10.532978+00:00 (updated-by): Updated: section:notes


## 2026-10-07 optimisation audit disposition

Current upstream master still has the UID fast-path gap: `ggml_cuda_graph_update_required()` returns false when non-zero `cgraph->uid == graph->uid` before its full node/source property comparison. Newer graph warmup/reset handling does not close that early return.

PRBE67 is the sole owner for this hypothesis; PRBE66 is a duplicate/fallback control, not a second implementation. Before patching, instrument the UID early return with node count and a compact Flash Attention K signature (`src[1]->ne[1]`, data pointer, strides) and reproduce the gfx1201 long-context failure against graphs-off.

If the failing transition does not combine the same UID with a changed FA signature, close this hypothesis. If confirmed, prototype only a changed-signature guard that falls through to the existing full `node_props` comparison. Do not add another fingerprint cache, graph map, scheduler or allocator. Use selective graphs-off only as a control/rollback.

Promotion requires repeated same-process multi-request, multi-ubatch, shallow/~80K/>=160K context, MTP depth controls where applicable, output parity versus graphs-off, zero stale replays, no extra recapture for stable shapes, and <=1% stable gfx1100/gfx1201 regression. Performance promotion requires CI95-low-positive >=3% where execution differs; correctness fixes may promote without a speed gain.

Upstream issue #23579 independently shows gfx1201 multi-GPU Flash Attention can fail under HIP graph capture and that graphs-off restores execution; it does not prove this UID mechanism. No hardware test or prototype was run by this audit.
