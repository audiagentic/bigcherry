---
id: PRBE101
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-12T09:24:56.124455+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
---

# Topology-gate RD42 shared-expert overlap: auto-disable under multi-GPU layer-split

## Description

Patch 1215's RD42 shared-expert stream overlap is currently a single global on/off switch (GGML_CUDA_GRAPH_OPT env var, read once via a static getenv() in a per-device cuda_ctx-scoped function in ggml-cuda.cu). Real hardware evidence (2026-09-12, see PRBE35/1215 README) found it gives a real +2.47% win on single-GPU gfx1100 but a real, consistent ~-4.4% REGRESSION under multi-GPU -sm layer split, and is flat/neutral under -sm tensor. Today the only way to avoid the layer-split regression is for the operator to remember not to set the env var in that deployment shape -- the code has no awareness of split topology at all.

## Steps



## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Filed 2026-09-12 directly from a user question during 1215/1216 real-hardware validation (PRBE35): 'are we able to gate these features based on topology? eg layer is clearly not a good win for this so when we run layer should it or can it be inactive.'

Real facts checked before filing:
- GGML_CUDA_GRAPH_OPT is read once via `static bool enable_graph_optimization = [] { const char* env = getenv("GGML_CUDA_GRAPH_OPT"); return env != nullptr && atoi(env) == 1; }();` inside ggml_cuda_graph_optimize (ggml-cuda.cu:4800), called per-device from each device's own cuda_ctx -- it has no visibility into how many OTHER devices are in play or what split mode is active.
- A real device-count accessor already exists elsewhere in the same file (`ggml_backend_cuda_get_device_count()`), but that returns total visible CUDA devices, not "devices this cgraph/model actually uses" or "split mode" -- neither is the same concept as what's needed here.
- Plausible (NOT YET CONFIRMED) mechanism for the layer-split regression: under -sm layer, each token's work is already serialized layer-by-layer across devices (each device hands off to the next via a copy at its layer boundary), leaving little genuine main-stream idle time for the auxiliary stream to exploit -- so the extra stream/event bookkeeping this patch adds becomes pure overhead rather than real overlap. This is a hypothesis by analogy to the gfx1100 single-GPU case (where real rocprofv3 profiling DID confirm genuine overlap) -- it has NOT been separately profiled for the layer-split case. Confirm via the same rocprofv3 methodology before designing the exact guard condition, since the guard should target the real cause (e.g. "is there large genuine main-stream idle time to overlap into"), not just "device count > 1" as a proxy that might be wrong.

Scope for whoever picks this up:
1. Profile the -sm layer regression case with rocprofv3 the same way the gfx1100 single-GPU win was profiled, to confirm/refute the serialization hypothesis above.
2. Design the real guard condition based on that evidence -- likely needs new plumbing to expose "is this graph's execution split across multiple devices in a way that already serializes main-stream work" to the per-device graph-optimize call site, not just a device-count check.
3. Change the activation condition from a pure env-var toggle to env-var-AND-topology-eligible, so `-sm layer` deployments get the patch's mechanism automatically inert without requiring the operator to remember to unset GGML_CUDA_GRAPH_OPT.
4. Re-run the full 2x2 (OFF/ON x layer/tensor/single-GPU) real hardware matrix afterward to confirm the new guard actually fixes the regression without disturbing the real single-GPU win.

This is downstream of PRBE35's existing real hardware evidence -- do not re-derive the base +2.47%/-4.4%/flat findings, they are already established there.

## Change Log

- 2026-09-12T09:24:56.124455+00:00 (created-by): Created by agent
