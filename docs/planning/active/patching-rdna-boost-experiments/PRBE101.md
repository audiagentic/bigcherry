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

TODO, corrected per GPT review (req_b43762f844fb40b3). The proposed guard was at the wrong scope/function: the verified `enable_graph_optimization` block (getenv("GGML_CUDA_GRAPH_OPT")) lives in `ggml_backend_cuda_graph_optimize` (confirmed at b11126 ggml-cuda.cu:4504-4609), NOT in `ggml_backend_cuda_graph_compute` (a separate function at line 4420, the actual graph-execution vtable callback). Returning at the TOP-LEVEL `if (!enable_graph_optimization || is_layer_split_member) return;` would disable upstream QKV-concurrency optimizations too, not just RD42's shared-expert overlap specifically -- the guard must be scoped to 1215's INSERTED shared-expert scan only, not the whole graph-optimize gate. Scheduler->CUDA-context plumbing (how is_layer_split_member gets set) remains unspecified.

## Steps

1. Leave the top-level `enable_graph_optimization` block in ggml_backend_cuda_graph_optimize (ggml-cuda.cu:4604-4609) UNCHANGED -- do not add the layer-split guard there, since that would also disable unrelated upstream QKV-concurrency optimizations this same function performs. 2. Add topology state to ggml_backend_cuda_context (e.g. an `ggml_cuda_execution_topology` struct per GPT's design: accelerator_count/multi_device/layer_split/row_or_tensor_split/rd42_slack enum{UNKNOWN,PROVEN_AVAILABLE,PROVEN_UNAVAILABLE}). 3. Expose a setter through ggml_backend_cuda_reg_get_proc_address (e.g. ggml_backend_cuda_set_execution_topology), invoked from src/llama-context.cpp after backend_ptrs assembly, using `model.n_devices() > 1 && model.split_mode() == LLAMA_SPLIT_MODE_LAYER`. 4. Gate ONLY 1215's inserted RD42 shared-expert scan (`for (int join_idx = 0; join_idx < cgraph->n_nodes; ++join_idx)` in patches/1215_rd394041_amd_stream_moe_overlap/patch.py) with `!(multi_device && layer_split)` -- not the surrounding function's top-level enable_graph_optimization check. 5. Set patch.toml requires=["1215_rd394041_amd_stream_moe_overlap","1216_rd43_concurrent_join_fusion_guard"]. 6. Run rocprofv3 profiling (per this plan's existing step 4) BEFORE finalizing the guard predicate -- profiling remains a validation prerequisite and must NOT leave the implementation predicate itself undecided once profiling is done.

## Detailed Solution & Technical Design

Profiling plan for step 4 (confirming the serialization hypothesis before hardcoding the guard): capture rocprofv3 traces of BOTH a single-GPU gfx1100 run (the case ALREADY profiled and confirmed to show genuine overlap, per the item's notes) and a `-sm layer` dual-GPU run, with GGML_CUDA_GRAPH_OPT=1 in both, and compare main-stream idle-time-during-shared-expert-window between the two. If layer-split truly serializes (each device already blocked waiting for the previous device's layer output before it can do anything, including the shared-expert branch), the aux-stream overlap window collapses to ~0 in the trace regardless of the stream-optimization machinery, confirming 'is_layer_split_member' is the RIGHT proxy (not just a convenient one). If overlap window is still non-trivial in the layer-split trace but the -4.4% regression persists anyway, the real cause is elsewhere (e.g. added stream/event bookkeeping overhead exceeding a smaller-than-single-GPU overlap benefit) and the guard condition needs to be more surgical (e.g. a minimum-overlap-window threshold rather than a binary split-mode flag).

## Code Samples & Guidance

Real b11126 anchor (ggml-cuda.cu, `ggml_backend_cuda_graph_compute`, confirmed this session at line ~4605):\n```cpp\nstatic bool enable_graph_optimization = [] {\n    const char * env     = getenv("GGML_CUDA_GRAPH_OPT");\n    return env != nullptr && atoi(env) == 1;\n}();\n\nif (!enable_graph_optimization) {\n    return;\n}\n```\nProposed change (sketch, pending the real scheduler-plumbing investigation in steps 1-2):\n```cpp\nstatic bool enable_graph_optimization = [] {\n    const char * env = getenv("GGML_CUDA_GRAPH_OPT");\n    return env != nullptr && atoi(env) == 1;\n}();\n\nif (!enable_graph_optimization || cuda_ctx->is_layer_split_member) {\n    return;\n}\n```\nThis mirrors patch 1217's existing pattern (env-var AND architecture-eligible) for env-var AND NOT-layer-split. patch.toml: id="<order>_rd42_topology_gate", state="untested", backend="hip", plan-item="PRBE101", requires=["1215_rd394041_amd_stream_moe_overlap", "1216_rd43_concurrent_join_fusion_guard"] (this guard only matters once RD42/RD43 are applied).

## Files

ggml/src/ggml-cuda/ggml-cuda.cu (ggml_backend_cuda_graph_compute activation site, ggml_backend_cuda_context struct in common.cuh for the new field), llama.cpp scheduler/model-loading layer (exact file TBD by grep for LLAMA_SPLIT_MODE_LAYER consumption -- likely src/llama-model.cpp or src/llama-context.cpp), patches/1261_nro10_spec_ctx_other_devices/ (read first, closest existing scheduler-to-backend-context plumbing analog), new package patches/<order>_rd42_topology_gate/ (requires 1215+1216).

## Validation

rocprofv3 profiling (step 4) is a genuine prerequisite to finalizing the guard condition, not just to validating it -- same pattern as PRBE49's reproduction-first gate. Offline: patch-lint, patch-rebase-check with 1215+1216 present. Hardware (Brutus, not run here): full 2x2 matrix (OFF/ON x layer-split/tensor-split/single-GPU) re-run after the guard change, confirming the -4.4% layer-split regression is gone while the +2.47% single-GPU win and tensor-split neutrality are preserved.

## Effort & Risk

M; the item itself flags the real risk -- device-count-as-proxy might be wrong, and the correct guard needs profiler evidence, not a guess. New scheduler-to-backend-context plumbing (step 1-2) also touches code outside ggml-cuda.cu, widening blast radius slightly beyond a pure ggml-cuda.cu change.

## Standards

Fail-closed: an unconfirmed guard condition must not be shipped as if it were profiler-validated; coordinate with patch 1217's existing env-var+architecture gating pattern rather than introducing a second, inconsistent gating convention.

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

2026-09-24 relevance at b11126: patch 1215 (RD42's actual implementation) confirmed present and read in full; the vulnerable activation site independently re-verified at its current (drifted) line number. Found a direct precedent (patch 1217/RD44) for extending this exact activation condition beyond pure env-var, which should inform this item's guard-condition design. GPT design request req_83fbfa0995034d2d (covering this + PRBE57/58/59) was in progress when this plan was authored; check for its response and merge/reconcile -- GPT was asked for a compact scaffolding design plus a rocprofv3 profiling sketch.

2026-09-24 GPT req_83fbfa0995034d2d COMPLETED, and closely converges with this plan's independently-authored design (both landed on: a per-backend-context topology struct set once at scheduler-init time, an env-var-AND-NOT-layer-split guard restricted specifically to the shared-expert overlap transform, and rocprofv3-based profiling before finalizing). GPT's version is more fleshed out: `ggml_cuda_execution_topology` struct (accelerator_count/multi_device/layer_split/row_or_tensor_split/rd42_slack enum {UNKNOWN,PROVEN_AVAILABLE,PROVEN_UNAVAILABLE}) embedded in ggml_backend_cuda_context, set via a new exposed backend proc `ggml_backend_cuda_set_execution_topology` called from llama_context after backend_ptrs assembly and before the first ggml_backend_sched_new()/reserve -- this is a concrete, real hook point this plan's step 1-2 only gestured at. Immediate guard: `enable_graph_optimization && !(multi_device && layer_split)` (matches this plan's sketch almost exactly). ROCTX marker scheme for the profiling step (RD42_REGION_BEGIN/ROUTED_BEGIN/SHARED_BEGIN/JOIN/REGION_END, computing main_idle_us vs actual_overlap_us) is more concrete than this plan's prose description and should be adopted directly. GPT explicitly cautions 'do not invent an online timing formula yet' -- rd42_slack starts UNKNOWN/conservative-disabled for layer-split until profiler evidence promotes it. Prefer GPT's struct/hook design when implementing; both designs agree on the core guard condition, which increases confidence it's correct.

2026-09-24 GPT review req_b43762f844fb40b3 applied: corrected the guard's target function from ggml_backend_cuda_graph_compute to the real host, ggml_backend_cuda_graph_optimize (verified at b11126 ggml-cuda.cu:4504); restricted the guard to only 1215's inserted RD42 shared-expert scan rather than the function's top-level enable_graph_optimization gate (which also covers unrelated upstream QKV-concurrency optimizations); specified the scheduler->CUDA-context topology plumbing via a new backend proc address setter.

## Change Log

- 2026-09-12T09:24:56.124455+00:00 (created-by): Created by agent
- 2026-09-24T04:53:18.095800+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:standards, section:notes
- 2026-09-24T04:53:56.397366+00:00 (updated-by): Updated: section:notes
- 2026-09-24T05:09:59.470605+00:00 (updated-by): Updated: section:description, section:steps, section:notes
