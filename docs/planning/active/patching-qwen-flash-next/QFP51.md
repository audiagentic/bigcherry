---
id: QFP51
order: 51
plan: patching-qwen-flash-next
state: done
created-at: '2026-10-10T21:23:38.446485+00:00'
breadth: ''
skill: advanced
created-by: claude
priority: P2
work: M
---

# Scheduler split-template cache — closed: reused graphs already skip re-splitting

## Authoritative disposition (2026-10-13; llama.cpp b11552)

**Done / reject new split-template cache.** The proposed MTP verify-width optimisation is already achieved by existing llama graph reuse: a reused graph does not call `ggml_backend_sched_split_graph`. Historical QFP51 asserted that every verification graph is split from scratch and conflated scheduler **compute-split input/copy time** with **split-graph construction**. The measured decode cost is real, but it is owned by QFP16/1325 and is not removable by caching a split plan. No patch, scheduler cache, dispatcher, allocator, configuration option, hardware queue, or performance gain is created here. The original proposal remains in Git history.

### Exact execution path and lifetime (pinned b11552 `23b0202a189c44a54625aadcb37a946dd1d6278`)

1. `src/llama-context.cpp::llama_context::process_ubatch` checks `gf_res_prev_active == res && res->can_reuse(gparams)`. **Reuse arm:** set live inputs and call `graph_compute`; it does **not** call `ggml_backend_sched_alloc_graph`. **Miss arm:** build the graph, call `ggml_backend_sched_alloc_graph(sched.get(), gf)`, then set inputs and compute. This is a *single previous-result* reuse, not a multi-width graph cache.
2. `ggml/src/ggml-backend.cpp::ggml_backend_sched_alloc_graph` calls `ggml_backend_sched_split_graph`, then `ggml_backend_sched_alloc_splits`, and sets `sched->is_alloc = true`. `ggml_backend_sched_graph_compute_async` only invokes allocation/splitting when `!sched->is_alloc`; otherwise it directly invokes `ggml_backend_sched_compute_splits`. `ggml_backend_sched_reserve` can also split its measurement graph during initial reservation; this is not per-token resplitting.
3. `ggml_backend_sched_compute_splits` still performs **per-call** input synchronization/copies and backend async compute. Patch `1325_sched_split_timing` instruments this execution loop, not `ggml_backend_sched_split_graph`. Patch `1319_process_ubatch_timing` separately reports graph build/reuse and graph-compute host time; `1320_meta_compute_timing` reports Meta subgraph launch and AllReduce enqueue. Do not relabel any of these as a measured split-construction cost.
4. `ggml_backend_sched_split_graph` recomputes backend placement from buffer/support predicates, frees/recreates `sched->ctx`, mutates `node->src[j]` to per-copy tensors, constructs split inputs and backend-specific `ggml_backend_graph_optimize` allocation dependencies, rebuilds the allocation graph, and assigns new split graph UIDs. A same-shape template cannot safely retain raw tensors, source pointers, allocation dependencies, backend IDs, `cur_copy`/events or UIDs across graph generations.
5. `ggml/src/ggml-backend-meta.cpp::ggml_backend_meta_graph_compute` already skips rebuilding its child subgraphs when `cgraph->uid == backend_ctx->uid`. Its UID cache is distinct from a scheduler split-template cache. Changing scheduler UIDs to force a hit could instead conceal changed tensor bindings. No-P2P Brutus placement makes input copy cost particularly relevant; it does not make the split-template proposal valid.

### Measured BigCherry evidence, correctly attributed

- QFP16 / patch 1319, Flash-Next target MTP verify near 10K: **98% graph reuse**; reused graph-build/reuse **0.003 ms** median, inputs **0.09 ms**, graph-compute host **5.01 ms**. The 5.01 ms includes compute-split execution, not a 5.01 ms re-split.
- QFP16 / patches 1320+1325 at the same approximate context: CPU split compute **0.27 ms**, Meta split **2.63 ms input handling** plus **2.23 ms compute submit**; the latter includes Meta subgraph launches/collective enqueue. These are diagnostic host timings, not end-to-end GPU gains.
- QFP51's 24K prefill observation: **0 of 76 prompt graphs reused**, with approximately **11.4 ms/batch** combined graph build/allocation and approximately **4 ms/batch** Meta rebuild. It does not isolate scheduler split time and gives no repeated same-signature prefill template opportunity. QFP50/1359 already reorders prompt work ahead of draft catch-up.
- Verify widths 2–8 have no isolated scheduler *split-construction* timing census. Repeated width alone does not imply a new split is executed: existing graph reuse must be checked first. No new benchmark or measured improvement in this audit.

### Ownership and non-duplication

- **QFP16 / 1319+1320+1325**: residual target verification host submit, synchronous split-input copies, Meta launch/AllReduce host costs. Existing diagnostics remain authoritative.
- **QFP50 / 1359**: prompt pipeline lookahead and fenced NextN lifetime; no additional pipeline work here.
- **QFP46**: host/device critical-path attribution and request/graph/event evidence; **QFP52**: NextN output ownership. **PRBE67/1356**: independent HIP graph UID/binding correctness. Do not modify or take ownership of any of these paths.
- Pinned and current upstream `ggml-backend.cpp` / `llama-context.cpp` retain the allocation-only split and reused-graph fast path (source checked 2026-10-13). [llama.cpp RPC #29287](https://github.com/ggml-org/llama.cpp/pull/29287) caches **remote serialized RPC graphs** by UID, not GGML scheduler split plans; its external APU throughput does not transfer to gfx1100/gfx1201. [SGLang static-buffer replay](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py) keys captured execution and manages persistent inputs, not this scheduler transformation. Neither should be ported.

### Bounded reopening gate (only if genuinely new evidence)

1. First use existing `BIGCHERRY_SUBMIT_TIMING` receipts to classify **graph reuse hit/miss**, phase, MTP verify width 2–8, prompt chunk, model/build pin, physical gfx1100/gfx1201/gfx1030 placement and the host critical path. Do **not** infer splitting from `BIGCHERRY_SCHED_SPLIT` input_us/compute_us. A miss-only `ggml_backend_sched_split_graph` span may be added inside QFP16 instrumentation *only if* existing graph_us data indicates material headroom; no new telemetry system.
2. Reopen only if two or more *new graphs* have a truly identical structural signature (op topology, edges, tensor types/shapes/strides, buffer ownership and generation, backend support/placement, graph flags, Meta child layout, allocation deps and copy slot), and **measured exposed** split-construction wall cost permits at least **3% end-to-end** improvement. If the same graph is already reused, or all prompt signatures differ, close without a hardware queue.
3. A future owner must rematerialize pointers and backend-specific dependencies, preserve copy/event generations, UID invalidation and allocator ownership, and fall back to native splitting on any mismatch. Never reuse `ggml_backend_sched_split` or its mutated `node->src[]` directly.
4. Only after the above gate: one default-off, same-binary A/B on Brutus with 8K/24K/98K and long context, verify widths 2–8, multi-request same-process, multi-ubatch, prompt-cache/cancel/context-shift, graph reallocation checks, logits/KLD/greedy/MTP acceptance and copy/work accounting. Require positive paired CI95, >=3% end-to-end gain, <=1% decode/control regression, no missing work, no extra VRAM/graph growth; otherwise reject.

### Validation actually performed

Pinned source assertions **16/16** passed; current upstream relevant-path checks **4/4** passed; host-only admission/fallback model **12/12** passed and Python compiled. These prove source predicates and a model of eligibility, **not** an executed GGML graph, HIP build, hardware test, or speedup. No repository pytest, patch application or GPU benchmark was run. No changes to active Radiance mailbox/expert-helper, patch-prefix, GPU lanes or queued experiments.

## Change log

- 2026-10-13: closed duplicate split-template optimisation; corrected attribution and ownership. Previous implementation proposal is superseded and retained in Git history.
