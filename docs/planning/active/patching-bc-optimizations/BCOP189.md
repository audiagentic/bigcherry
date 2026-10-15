---
id: BCOP189
order: 189
plan: patching-bc-optimizations
state: done
created-at: '2026-10-13T16:07:00Z'
created-by: agent
priority: P2
work: S
---

# QFP51: retire redundant scheduler split-template cache

**Discovery/change:** Pinned b11552 `llama_context::process_ubatch` reuses the previous graph without `ggml_backend_sched_alloc_graph`; `ggml_backend_sched_graph_compute_async` calls split/allocate only when `!sched->is_alloc`. QFP51 mistook patch 1325's per-call `compute_splits` input/copy timing for graph re-splitting. Existing target MTP verification already reports 98% graph reuse (1319), while 24K prefill had 0/76 reusable graphs. No measured residual split-construction bottleneck or QFP51 speedup.

**Owner/consolidation:** QFP51 marked `done` with no template cache/patch. QFP16 owns split-input and Meta submit residuals through 1319/1320/1325; QFP50/1359 owns prefill lookahead; QFP46 owns critical-path evidence; QFP52 NextN placement; PRBE67 UID safety. Active Radiance GPU-mailbox/helper work excluded and untouched. Upstream RPC #29287 and SGLang static replay are not interchangeable scheduler mechanisms.

**Unresolved action / terminal gate:** None for QFP51. Reopen only with a correlated same-signature **new-graph** miss and an isolated exposed `split_graph` critical-path ceiling >=3% end-to-end; reject reused graphs and unique-shape prompts. Then require safe rematerialization of backend bindings, allocation dependencies, copy generations and UIDs before any AMD A/B. Otherwise terminal `done`.

**Evidence:** llama.cpp b11552 `23b0202a`, current upstream `ggml-backend.cpp`/`llama-context.cpp`; QFP16 receipts 1319/1320/1325, QFP50/1359; [RPC #29287](https://github.com/ggml-org/llama.cpp/pull/29287). Pinned source 16/16, current upstream 4/4, host-only fixture 12/12, Python compile passed. No repository pytest, HIP compile, GPU run, or new performance measurement.
