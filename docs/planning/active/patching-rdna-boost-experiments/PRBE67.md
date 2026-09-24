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
priority: null
---

# HIP-GRAPH-002: Force graph recapture when FA stream topology changes

## Description

TODO, primary approach (per item's own preference order). Confirmed real gap at b11126: `ggml_cuda_graph_get_key(ggml_cgraph * cgraph)` in ggml/src/ggml-cuda/ggml-cuda.cu returns only `cgraph->nodes[0]` (the first node's pointer) as the cache key for `cuda_ctx->cuda_graph(graph_key)` -- it carries NO shape/topology information, so a captured graph is reused across calls whose FA stream/shape topology differs as long as the first node's address is the same. This is a plausible root cause for R9700/gfx1201 100K+-context FA instability: the graph executable captured for one topology gets blindly `cudaGraphExecUpdate`'d or replayed against a structurally different one.

TODO, premise corrected (same underlying disproof as PRBE66). GPT confirmed the prior claim that `ggml_cuda_graph_get_key` returning only `cgraph->nodes[0]` means 'no topology information' is misleading: the caller, `ggml_cuda_graph_update_required` (verified, ggml-cuda.cu ~2591-2635), ALREADY does full per-node/per-source property comparison (memcmp of copied tensor struct plus each source's data ptr/ne[]/nb[]) and forces recapture on any difference -- this plan's proposed FA-only fingerprint (`ggml_cuda_graph_topology_fingerprint`) would DUPLICATE and be WEAKER than what already exists (it only hashes FA nodes' ne[], missing the existing check's src-pointer and full-tensor-struct comparison). The only plausible uncovered case is the early `cgraph->uid != 0 && cgraph->uid == graph->uid` fast path (same line as PRBE66's finding), which skips the property comparison entirely when the cgraph object's own uid is unchanged.

## Steps

1. Reproduce the R9700 100K+ FA failure with GGML_CUDA_OP_TIMING/BIGCHERRY_PATCH_TRACE-style logging added around `ggml_cuda_graph_update_required` and `ggml_cuda_graph_get_key` to confirm: does the key (first-node pointer) stay identical across the failing transition while the actual node count/shapes change underneath it?
2. If confirmed: extend the key/identity check in `ggml_cuda_graph_update_required` (ggml-cuda.cu, the function that calls `ggml_cuda_graph_get_key` and compares against the cached `ggml_cuda_graph`) to also compare a cheap topology fingerprint -- e.g. `cgraph->n_nodes` plus a hash of each node's `ne[]`/`op` for FA-family ops only (avoid full-graph hashing cost) -- forcing recapture (`ggml_cuda_graph_update_required` returning true, or evicting the cached entry) whenever the fingerprint differs from the one stored alongside the cached graph, even if `nodes[0]` is unchanged.
3. If NOT confirmed by step 1 (key already effectively distinguishes the failing case some other way, or the bug is elsewhere e.g. in `cudaGraphExecUpdate`'s error handling at ggml-cuda.cu ~line 2641-2657), stop and report the real mechanism found instead of proceeding on the wrong hypothesis -- do not force this design onto an unconfirmed cause.
4. Add the new topology-fingerprint field to the cached `ggml_cuda_graph` struct (ggml-cuda.cu, near its other members) and thread it through `cuda_ctx->cuda_graph(first_node_ptr)`'s map value.
5. Correctness/regression test: repeated 100K+-context runs (long-run stability, no crash, output-identical vs graphs-disabled baseline) plus stable-topology controls (short context, repeated identical shape) confirming recapture does NOT fire spuriously and the existing graph speedup is retained.
6. Compare recapture frequency/cost directly against PRBE66's narrower bypass approach; promote this over PRBE66 only if recapture cost stays bounded and output-identical across repeated long-context runs.

1. Do NOT implement the `ggml_cuda_graph_topology_fingerprint()` design from the prior plan draft -- it duplicates and is weaker than the existing full node-property comparison already in `ggml_cuda_graph_update_required`.
2. Instrument the R9700/gfx1201 100K+-context FA failure specifically at the `cgraph->uid == graph->uid` fast-path branch (same instrumentation as PRBE66 step 1 -- this item and PRBE66 must NOT ship two separate, possibly-conflicting fixes to the same function; coordinate and converge on one before either is implemented) to determine: does the failing transition hit this fast path with `uid` unchanged but real node/source data changed underneath?
3. If confirmed: the correct fix is not a NEW fingerprint mechanism but closing the fast-path gap itself -- either stop trusting `uid` equality alone (always run the existing full property comparison, accepting its cost), or add a narrow, cheap FA-specific check (src[1]->ne[1] for FLASH_ATTN_EXT nodes) that must ALSO match before the fast path is allowed to return false.
4. If NOT confirmed (uid fast path is not implicated): report the actual mechanism found (e.g. `cudaGraphExecUpdate`'s error handling at ggml-cuda.cu ~2641-2657, already flagged as a fallback hypothesis in the original plan) rather than proceeding on this design.
5. This item and PRBE66 converge on a single implementation once step 2's evidence is in hand -- do not implement both independently.
6. Correctness/regression test: repeated 100K+-context runs (no crash, output-identical to graphs-disabled baseline) plus stable-topology controls confirming the fast path still short-circuits when nothing changed (no spurious recapture cost, existing speedup retained).

## Detailed Solution & Technical Design

`ggml_cuda_graph_get_key` intentionally uses a cheap identity (first node pointer) as a proxy for graph structure, valid only because llama.cpp usually reuses the same cgraph object across identical-shape calls. Long-context FA appears to violate that assumption (same first node, different downstream topology as sequence length crosses chunk boundaries or KV cache layout changes). The fix adds a topology fingerprint stored alongside the cached `ggml_cuda_graph` and checked in `ggml_cuda_graph_update_required`, so a real structural change forces recapture even when the first-node identity is stable. This must stay independent of `GGML_CUDA_GRAPH_OPT` (patch 1215/RD42's separate stream-overlap toggle) -- it operates on the base graph-capture mechanism, not the optimization pass.

## Code Samples & Guidance

Confirmed anchors at b11126, ggml/src/ggml-cuda/ggml-cuda.cu:
```cpp
static const void * ggml_cuda_graph_get_key(ggml_cgraph * cgraph) {
    return cgraph->nodes[0];
}

static bool ggml_cuda_graph_update_required(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph) {
    bool res = false;

    const void * graph_key = ggml_cuda_graph_get_key(cgraph);
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
    ...
```
Design sketch (exact anchor/replacement must be re-verified against the FULL body of `ggml_cuda_graph_update_required` -- only its first lines are shown above; read the rest with `git -C work/upstream/llama.cpp.git show b11126:ggml/src/ggml-cuda/ggml-cuda.cu | sed -n '2595,2645p'` before writing the final patch.py):
```cpp
// add near the key function:
static uint64_t ggml_cuda_graph_topology_fingerprint(ggml_cgraph * cgraph) {
    uint64_t h = (uint64_t) cgraph->n_nodes;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * t = cgraph->nodes[i];
        if (t->op != GGML_OP_FLASH_ATTN_EXT) continue;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) h = h * 1000003 ^ (uint64_t) t->ne[d];
    }
    return h;
}
```
followed by storing/comparing this alongside the cached graph in `ggml_cuda_graph_update_required` (exact insertion point TBD pending the full function body read above).
Patch package sketch: `patches/12xx_hip_graph_fa_topology_recapture/patch.toml` (backend="hip", state="untested", experiment-contracts=["RD84-HIP-GRAPH-RECAPTURE"]).

## Files

ggml/src/ggml-cuda/ggml-cuda.cu; new instrumentation for step 1 (temporary, qualification-only, see PRBE71's ablation-knob convention); patches/12xx_hip_graph_fa_topology_recapture/{patch.toml,patch.py,SUMMARY.md}.

## Validation

PYTHONPATH=tools python -m bigcherry patch-lint; patch-rebase-check. Hardware (not run here): repeated R9700/gfx1201 100K+-context FA runs must not crash and must be output-identical to a graphs-disabled reference; stable-topology short-context runs must show no extra recapture events; PP/TG and recapture-event-count telemetry compared against PRBE66's bypass and against ungated baseline via python -m bigcherry.patch.validation_campaign.

## Effort & Risk

M; risk is fingerprint cost/collision and getting the exact insertion point wrong without reading the full update_required body first (explicitly flagged as a required pre-step, not assumed here).

## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Promote over PRBE66 only when repeated long-context runs are robust, output-identical, and recapture cost is bounded while stable graphs retain their speedup.

## Notes

Supersedes: RD84
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd84

2026-09-24 relevance at b11126: TODO confirmed with a real, verified gap -- `ggml_cuda_graph_get_key` returns only `cgraph->nodes[0]` (git show b11126:ggml/src/ggml-cuda/ggml-cuda.cu line 2591-2593), no shape/topology awareness at all, supporting the stale-key hypothesis as plausible root cause (still unconfirmed pending step 1's real reproduction). GPT design request: gateway rejected all submissions this session (VAL-AGW-025 / EXT-GPTAUTO-003); plan authored directly from verified source -- no GPT request id.

2026-09-24 GPT review req_d55aed71224e43a8 applied: NOT-READY -- disproved 'no shape/topology awareness' premise (same finding as PRBE66); the proposed FA-only fingerprint duplicates and is weaker than the existing full node-property comparison in ggml_cuda_graph_update_required; narrowed to instrumenting and closing the uid-equality fast-path gap, converging with PRBE66 on one fix rather than two.

## Change Log

- 2026-09-09T10:58:14.009489+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:15:20.253208+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

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
