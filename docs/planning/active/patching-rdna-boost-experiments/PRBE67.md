---
id: PRBE67
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:58:14.009489+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: P2
---

# HIP graph UID reuse: qualify changed bindings before recapture work

## 2026-10-13 authoritative audit (llama.cpp b11552; supersedes older fingerprint/recapture proposals)

**Disposition: pending, correctness-only, no implementation or GPU job authorized without a correlated same-UID property mismatch.** The old proposed FlashAttention-only topology fingerprint is retired. A graph key based on the first node is not, by itself, evidence of stale replay: the existing slow path already compares full node structs and source pointers, shapes and strides. The unproven risk is the nonzero **UID equality fast return**, which skips that comparison. Do not infer that it caused the historical R9700 100K+ instability or the separate 1356 worker race.

### Exact implementation / lifetime map

- Pinned and current upstream `ggml/src/ggml-cuda/ggml-cuda.cu::ggml_cuda_graph_get_key` use `cgraph->nodes[0]` as the per-backend-context graph-map key. `ggml_cuda_graph_update_required` returns **false** when `cgraph->uid != 0 && cgraph->uid == graph->uid`, asserting only node count, *before* the slow `ggml_tensor` + `src[j]->data/ne/nb` comparison. If UID differs, it checks and updates `graph->node_props`. The subsequent `ggml_cuda_graph_update_executable` already re-instantiates on `cudaErrorGraphExecUpdateFailure`. Preserve this native fallback; do not create a second graph map or fingerprint registry.
- `ggml/src/ggml-backend-meta.cpp::ggml_backend_meta_graph_compute` uses `cgraph->uid == backend_ctx->uid` to skip subgraph rebuild, and assigns `cgraph_ij->uid = ggml_graph_next_uid()` on rebuild. Thus a reused Meta subgraph UID is normal. 1340's graph-owned arena may bind data before backend graph submission; **whether a live UID ever survives a changed binding is not established**. A different allocation base with the same tensor-relative offset would be caught by the native slow property comparison, but not by UID equality.
- `engines/llamacpp/patches/1356_meta_dispatch_workers/patch.py` currently adds a global shared/exclusive graph-compute lock and optional `BIGCHERRY_META_DISPATCH_GRAPH_TRACE` diagnostics; it **does not currently bypass the UID shortcut**. The earlier #159 UID-bypass attempt did not establish correctness (3/12 wrong greedy runs), and the subsequent lock attempt also failed (5/12); 1356 is evaluated/shelved, not a production fix. These experiments do not prove or disprove the single-context R9700 hypothesis.
- 1356's `bc_1356_graph_property_hash` intentionally normalizes a data pointer to `data - buffer_base` and hashes buffer name/size, not the absolute base. Two otherwise identical bindings at different bases therefore produce the **same diagnostic digest** even though native `node_props` would differ. The trace is a useful cross-process signature, **not** an authoritative same-process stale-replay detector. Do not treat equal `props` hashes as proof of safe replay.

### Ownership and exclusions

PRBE67 alone owns the **UID-fast-path property-admission hypothesis** and its bounded diagnostic. The historical PRBE66 alternative is a duplicate, not another patch. QFP41/1356 owns its shelved worker-mode race and diagnostic implementation; 1340/MSM02 owns Meta allocation/binding; 1302 owns graph OOM eviction; the native CUDA/HIP graph code owns capture/update. Reuse the existing comparison and QFP41 trace format where safe, not another graph scheduler, allocator, telemetry registry or graph cache. Recent Radiance GPU-mailbox, expert-helper, RAD01 and release work was active in the preceding 12 hours and is untouched; no hardware work is queued against those lanes.

### Gate 0 — cheapest host/static discriminator (completed in this audit)

The source-shaped host fixture `prbe67_uid_rebind_discriminator.py` modeled: same UID + changed buffer base; same UID + changed shape; changed/zero UID; changed node count; native slow comparison; and 1356's buffer-relative diagnostic. **10/10 host tests and Python compilation passed.** Actual pinned/master source assertions **13/13 passed**. These establish a *possible admission blind spot*, not a reproduced HIP failure. No repository pytest, C++/HIP compilation, patch application or GPU benchmark ran.

### Gate 1 — bounded live attribution, no production change

Only after hardware is idle, instrument **at the early-return decision** in `ggml_cuda_graph_update_required`, not after `cuda_graph_update_required` has already been decided in `ggml_backend_cuda_graph_compute`. Under a default-off diagnostic flag, compute the **same full node/source property comparison as the native slow path** before allowing a same-UID fast return. Log per physical GPU/backend context: source pin and build ID, graph key/UID, node count, first differing node/op/source index, old/new absolute data pointers (or process-local collision-free IDs), buffer owner/generation, shape/strides, graph update/capture/replay decision, request/ubatch/context and MTP depth. Keep normalized cross-process hashes only as secondary correlation. Do not mutate graph state or add device synchronizations in the diagnostic control. Never use diagnostic timing as a throughput result.

Require a matched graphs-on / graphs-off reproduction of the historical **single-R9700 >=100K-context FA** failure; separately classify the 1356 **dual-XTX 98K MTP+worker** symptom as an external control, not evidence that the same cause applies. Check repeated requests in one process, alternating ubatches, 8K/~80K/>=160K context, graph recapture/replay, and zero/nonzero UID. A changed UID, different first-node key, no same-UID property mismatch, or first divergence before FA/graph replay **rejects this UID hypothesis**; do not broaden into generic scheduler debugging.

### Gate 2 — only if a causal same-UID mismatch is observed

Prototype one **default-off** guard at the existing UID early return: on a proven susceptible graph, fall through to the existing full `node_props` comparison instead of returning false. Pseudocode:

```cpp
if (cgraph->uid != 0 && cgraph->uid == graph->uid) {
    if (!bc_uid_guard_qualified_for_this_graph) {
        return false; // existing default
    }
    // fall through to existing node/source property comparison
}
```

No separate FA-only hash, global graph recapture, new cache key, or automatic graph-disable policy. Test changed buffer bases with unchanged offsets, unchanged buffers, changed node/source shapes/strides, graph resize, allocator reuse, cancellation, concurrent Meta ranks, multi-request and MTP on/off. Prove no skipped graph work, correct full logits/KLD/greedy tokens versus graphs-off, memory safety and stable recapture counts. The diagnostic must distinguish `UID_MISMATCH_CAUSAL`, `UID_MISMATCH_NONCAUSAL`, `NO_UID_MISMATCH`, `NOT_REPRODUCED`, and `UNATTRIBUTED`.

### Promotion / terminal gate

Require the causal error to disappear under the opt-in guard in **>=12 repeated 98K/long-context runs** with multiple process starts and request/ubatch shifts, with graphs-on/off parity and zero wrong greedy outputs. Measure the guard's per-token CPU comparison cost and extra captures; stable-topology TG regression must be <=1% with a non-negative end-to-end correctness/performance tradeoff. If no causal mismatch is captured, close `UID_HYPOTHESIS_NOT_CONFIRMED` without code. If corrected but >1% control regression, retain a narrowly scoped safe fallback (graphs-off for the affected lane) and do not globally promote. Do not claim a speedup from a correctness-only fix.

### Upstream / fork mechanisms (checked 2026-10-13)

- Upstream llama.cpp **b11552 and master** retain the same UID early return and slow property comparison. [RPC #29287](https://github.com/ggml-org/llama.cpp/pull/29287) is **draft** and implements an RPC multi-graph UID cache, not a HIP graph replay fix; its buffer-free/UID-reuse test is a useful lifetime invariant only. **Do not port** the RPC cache.
- [SGLang dynamic replay contract #32432](https://github.com/sgl-project/sglang/issues/32432) and [request-slot geometry failure #34384](https://github.com/sgl-project/sglang/issues/34384) motivate explicit metadata/slot compatibility, but do not prove this BigCherry failure. [vLLM CUDA graph utilities](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/cudagraph_utils.py) synchronize prior offloader copies before replay; reuse the ownership invariant, not the Python graph framework.
- RDNA3 gfx1100, RDNA4 gfx1201 and gfx1030/no-P2P PCIe constraints affect scheduling and transport, **not** the UID equality rule itself. Never transfer NVIDIA/RPC results into AMD performance claims.

Historical 2026-09 and 2026-10-07 drafts, including contradictory fingerprint steps, are preserved in Git history, not executable instructions in this plan.
