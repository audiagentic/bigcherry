---
id: QFP53
order: 0
plan: patching-qwen-flash-next
state: superseded
created-at: '2026-10-11T21:08:26.013082+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Prompt graph reuse via coarse KV/k-pool padding — rejected; residual batch gap owned by QFP48

## Terminal disposition (2026-10-14, BCOP206)

**SUPERSEDED / NO_GAIN.** No remaining QFP53 implementation or hardware lane. Patch `1362_kv_pad_prompt` is **rejected** in both `patch.py` and `patch.toml`; its `SUMMARY.md` contains the hardware rejection. Do not requeue `kv-pad-prompt`, add a second graph cache, or implement speculative candidate B (building the next graph ahead): validated `1359_prefill_pipeline` already queues the next batch while the previous one executes. Move remaining unexplained prompt-boundary latency to QFP48's existing trace-led attribution. QFP50/1359 owns queue-ahead; QFP16/1326 and QFP27 own input-copy lifetime; QFP46 owns the dependency/graph-lifetime contract.

## What was actually measured (historical b11552, not new results)

- Initial 8K-depth timing: 40/40 prompt submits reported `reused=0` with KV-only padding because Qwen4Exp `llm_graph_input_kpool::can_reuse` also compares `pool_cells->ne[0]` to `get_n_kpool()`. `llama_memory_hybrid_idx_context::kpool_pad` advanced in 64-pool buckets. This was repaired inside 1362 by deriving pool padding from the same KV-cell quantum.
- After that fix (build `b-kvpad2`, N=4096), reuse occurred on **24/40** full 512-token submits. Mean graph-build host time dropped **7.1 -> 3.0 ms/batch**; the input step rose **3.3 -> 17.0 ms/batch** because reused graph input storage cannot be overwritten until its previous consumer retires.
- Same-binary Flash-Next ABBA: at ~8K, off warm 1443.7 t/s versus on 1436.3/1445.8 t/s; at ~24K, off 1465.5/1464.6 versus on 1466.2/1467.6 t/s. Greedy text matched. No meaningful prefill gain demonstrated; no CI or new GPU run in this audit.
- Dense Qwen3.8-27B Q8_0 dual-XTX with KV-only 1362 was ~0.4% slower at 10K and ~1.0% slower at 32K; at 32K padded and unpadded greedy outputs differed. The longer masked attention traversal changes floating-point reduction order. Do **not** claim universal bit identity.
- The original attribution of ~20-21 ms of host graph-build/Meta-rebuild/input work per batch as fully serial was incorrect. Under 1359 queue-ahead much of that host work overlaps the previous GPU batch. Kernel trace still shows ~6-7 gaps of ~3 ms near batch-boundary copies; the **exposed** cause and removable share remain unproven.

## Implementation trace and why this approach loses

1. `src/llama-kv-cache.cpp::llama_kv_cache::get_n_kv` pads the attention dimension; `src/llama-memory-hybrid-idx.cpp::kpool_pad` pads QSA pool count. Patch 1362 changes both and preserves masked/dead rows. Qwen4Exp's `llm_graph_input_kpool::can_reuse` checks pool width and KV width; changing only one cannot establish reuse.
2. `src/llama-context.cpp::llama_context::decode` and `src/llama-graph.cpp` reuse a graph only when input/shape ownership permits. The existing input buffers are not independent per in-flight generation. Reuse can therefore shift time from graph construction into waiting before `process_ubatch` writes the next input.
3. `ggml/src/ggml-backend-meta.cpp::ggml_backend_meta_graph_compute` rebuilds subgraphs when the graph UID changes, but its host `rebuild_us`/`launch_us` are not necessarily exposed device idle. `1320_meta_compute_timing` logs them under `BIGCHERRY_SUBMIT_TIMING`; do not sum host enqueue and device kernel time as serial.
4. `1359_prefill_pipeline` queues batch k+1 before collecting k's fenced NextN; two pinned host slots and backend events bound output lifetime. This already implements the useful part of candidate B without another scheduler or graph cache. Forcing graph reuse removes some host build work but can defeat that overlap.

## Current upstream and external mechanism decision

- Upstream llama.cpp PR #29958 (merged 2026-10-05) fixes graph-shape dependence on mutable hybrid k-pool `cache_safe`; it is a correctness baseline, **not** proof coarse KV padding helps.
- Upstream llama.cpp PR #30453 (merged 2026-10-13, after b11552) recreates **max_subgraphs** at **max_nnodes** when Meta arena is reallocated, instead of only current `n_subgraphs`/`cgraph->n_nodes`. Any future persistent/multi-shape graph proposal must first inherit/qualify that allocation invariant. Do not expand QFP53 to port it; QFP46/Meta owner handles shared graph correctness.
- vLLM's `CUDAGraphWrapper` deliberately delegates stable input storage to the caller and checks replay input addresses. It is an ownership precedent, not a portable AMD performance result; a second BigCherry graph-cache/allocator is unjustified.

## Residual admission belongs to QFP48, not QFP53

Use existing `tools/lab/flash-next/{kernel-step-census.py,gap-api-calls.py}` and 1319/1320/1359 telemetry. On a warmed same-binary production server at 8K/24K/98K, correlate **per request/chunk/rank**: previous final kernel/event, each HIP copy/upload and synchronize, first kernel of the next batch, `graph_uid`, Meta rebuild, input-generation reuse, and the draft fence. Do not subtract host and device timestamps from uncorrelated clocks. Compare 1359 on/off only as a diagnostic control, not as a proposed production regression.

At ~1465 t/s and 512 tokens, one batch takes ~349.5 ms; a +3% throughput target needs **at least ~10.18 ms of genuinely removable exposed wall time per batch** (`T*(1-1/1.03)`). The old ~21 ms gaps are only an unqualified ceiling until a joined timeline proves they are on the critical path and removable. If exposed opportunity is <3%, close without code. If >=3%, QFP48 must name one existing copy/wait owner, one default-off local intervention and an event/lifetime test before hardware qualification. Require two slots, multi-ubatch, long context, cancellation, context shift, deterministic output and MTP acceptance. Promote only with CI95-low >=3% end-to-end prefill/TTFT and <=1% decode/control regression.

## Validation for this audit

Repository metadata/source consistency checks 10/10 passed; five standard-library synthetic critical-path tests passed and compiled. The synthetic fixture only demonstrates why a hidden graph build is not a wall-time saving and shared input can introduce a wait; it is **not** a GPU timing or correctness test. No patch apply, pytest, HIP build or hardware run was performed.

## References

- BigCherry 1362 `patch.py`, `patch.toml`, `SUMMARY.md`; 1359 `patch.py`/`SUMMARY.md`; 1320 `patch.py`; QFP48, QFP46, QFP27.
- https://github.com/ggml-org/llama.cpp/pull/29958
- https://github.com/ggml-org/llama.cpp/pull/30453
- https://github.com/vllm-project/vllm/blob/main/vllm/compilation/cuda_graph.py

## Change log

- 2026-10-14: BCOP206 reconciles rejected 1362 with the QFP53 active-plan status and stale README; residual boundary diagnosis assigned to QFP48.