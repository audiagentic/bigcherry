---
id: MSM02
order: 2
plan: patching-meta-split-memory
state: pending
created-at: '2026-10-06T11:48:08.966158+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Per-device compute arena in the tensor-split backend

## Description

ggml_backend_meta_buffer_type_alloc_buffer() allocates the scheduler's compute arena at the same full size on every simple device (verified at pin b11402, ggml-backend-meta.cpp L1697-1720): 285 MiB at ctx 49152 and 1020 MiB at 245760 on each of the three cards, whatever share of the graph the card computes. Weights and KV already follow the split (alloc_buffer_n). Goal: each card allocates only what its own slices of the compute graph need, so a card with no attention share (the R9700 in the owner's layout) does not pay for context-shaped attention intermediates.

## Steps

FIRST STEP (new patch, flag off by default): add the per-device allocator and helper behind BIGCHERRY_META_PER_DEVICE_ARENA=1 with the common-size path untouched when the flag is off; report per-device arena sizes through 1339's BIGCHERRY_META_MEM. Then: offline mechanics test; build; identical greedy text and fidelity against the common-arena build; measure per-device MiB at ctx 49152 and 245760 for production and the owner's layout; only then consider default-on.

## Detailed Solution & Technical Design

Design from GPT (req_cc4f6bc692c2431f; its statements about the current code are verified, the design is not implemented). Rejected: scaling each device's buffer, or planning each tensor at its largest slice - ggml-alloc keeps ONE buffer_address per logical tensor (ggml_gallocr_allocate_node L623+, one high-water mark in ggml_gallocr_reserve_n_impl L825+), so every device still receives the same high-water mark and offsets contain holes; arbitrary compaction breaks views (init_tensor_impl L1224-1268), in-place aliasing and reuse. Recommended: a per-simple-device gallocr over the transformed per-device tensors. (1) one ggml_gallocr_t compute_galloc per backend_config in ggml_backend_meta_context; static alloc_buffer_n() unchanged. (2) the compute path of ggml_backend_meta_buffer_type_alloc_buffer() creates null/dummy per-device holders while the Meta buffer keeps its logical size. (3) ggml_backend_meta_buffer_init_tensor_impl(): with a null compute buffer create shape/src/view metadata only, leave buffer/data null. (4) new ggml_backend_meta_alloc_graph(meta_backend, cgraph): for each simple device map every Meta tensor of cgraph to its simple tensor, leave non-Meta tensors external, then ggml_gallocr_reserve + ggml_gallocr_alloc_graph on that device's graph - views, in-place reuse and different slice sizes are handled by the existing allocator. (5) call it from ggml_backend_sched_alloc_splits() right after the successful ggml_gallocr_alloc_graph() (L1793) and before anything can tensor_set (ggml_backend_meta_buffer_set_tensor L1426 needs real simple buffers); ggml_backend_sched_reserve() (L2131) must materialise the measure graph and run the helper before sched_reset() so load-time reservation is real. AllReduce scratch stays separate (ggml_backend_meta_graph_compute, max_tmp_size, L2240).

## Code Samples & Guidance



## Files

new patch package patches/1340_meta_per_device_arena/ (patch.py, patch.toml, SUMMARY.md), tools/tests/patch/test_1340_meta_per_device_arena.py; vendor files touched by the patch: ggml/src/ggml-backend-meta.cpp, ggml/src/ggml-backend.cpp

## Validation

Flag off: byte-identical behaviour (same greedy md5 as production). Flag on: same greedy text and probe distributions as flag off (the arena layout must not change any result), no OOM / illegal access under BIGCHERRY_PATCH_TRACE, per-device arena MiB from MSM01's report lower on the devices with smaller shares, peak VRAM lower in rocm-smi. Must compose with 1283 (expert split), 1303 (attention split) and 1326 (async host inputs).

## Effort & Risk

High: touches the scheduler's allocation path and the meta buffer's tensor initialisation; a wrong offset corrupts memory silently. Mitigation: flag off by default, identical-output gate, small steps.

## Standards

Package-only patch, fail-closed anchors, guards, expect_matches, offline test (bigcherry-patch-author). No legacy/back-compat shims: the flag is an off switch during qualification, not a second permanent path.

## Acceptance Criteria



## Notes

Base framework change (memory layout), not an RD enhancement: the gate is 'no regression + identical output + measured memory reduction'. Owner direction 2026-10-06: memory allocation should follow the split.

2026-10-06 FIRST STEP ON HARDWARE (1340_meta_per_device_arena by GPT, commits 595f01f2 + d59d58b2; build b-metamem-b11402b2). Flag off: unchanged (arms P, O, Pm, Om run and match). BIGCHERRY_META_PER_DEVICE_ARENA=1: the server dies silently during load in all four configurations (production and owner's layout, ctx 49152 and 245760); the log ends after 'BIGCHERRY_META_MEM arena dev=0 ... 267.34' and 'dev=1 ... 267.34' (880.8 at 245K) with no line for dev=2 - the crash is in or right after the third device (R9700), which has zero-sized slices for every attention-side tensor. The two arenas that were created are smaller than the common arena (267.3 vs 285.3 MiB at 49K, 880.8 vs 1020.9 at 245K). Offline test fixed (it sliced the forward declaration). Crash handed back to GPT with the evidence (req_c2487aeea618417f); fix goes inside 1340.

2026-10-06 ROOT CAUSE / FINAL SOURCE FIX: b11402 `ggml/src/ggml-cuda/fattn.cu::ggml_cuda_get_best_fattn_kernel()` computes `Q->ne[2] / K->ne[2]`; on the zero-attention-share R9700 the transformed FLASH_ATTN_EXT K slice has `ne[2] == 0`, and the per-device gallocr reached this backend alloc-size hook during reserve before the dev2 arena log. 1340 now keeps only zero-sized, bufferless, non-view transformed tensors external to the simple gallocr after asserting their COMPUTE flag is clear, using the Meta logical address solely as an allocator sentinel. CUDA/HIP skips COMPUTE-clear nodes, so the sentinel is never dereferenced; nonzero disabled outputs needed by AllReduce are still normally allocated. Final source fix: `de7ec9cab40c1c080335a9b37bdeabb9473727be`. A pinned mechanics regression now records the offending b11402 division and proves the sentinel is installed before `ggml_gallocr_reserve()`. Final build/fidelity/memory rerun remains the qualification gate.

2026-10-07 1340 RUNS END TO END after three fixes inside the patch (all on the zero-attention-share device, found with gdb): (1) zero-sized STATIC slices (dummy buffer, no data) were handed to the per-device gallocr -> GGML_ASSERT(tensor->buffer == NULL) in ggml_backend_tensor_alloc; now every zero-sized slice is external (f212d6aa). (2) the reserve path instantiates a graph without computing it, and repeated reserves filled a static buffer's simple-tensor view container -> ggml.c GGML_ASSERT(obj_new); the reserve now rotates the containers as a compute would (3b1e58fa). (3) GPT's guard that zero-sized deferred tensors are never COMPUTE nodes fired - they exist on a device with no attention share; guard removed (91137e22). RESULT (build b-metamem-b11402f2): arena MiB XTX / XTX / R9700: ctx 49152 267.3 / 267.3 / 225.2 against 285.3 common; ctx 245760 880.8 / 880.8 / 862.8 against 1020.9; with 1341 also on the R9700 arena is 210.7 and 618.8. Owner's layout (expert split, dense on the XTXs): greedy text identical with the flag on. PRODUCTION ROW SPLIT: NOT identical - greedy md5 06a46fbc vs f80ea4df, probes top-1 23/24, TV mean 0.072 max 0.221 (flag off against itself: 0.0000); speed equal (prefill 986.8/1052.7 vs 947.5/1067.1, decode 84.3/85.3 vs 84.4/86.5). OPEN DEFECT: a memory-layout change must not alter results. Working hypothesis (not verified): per-device allocators make different in-place / reuse decisions on devices with unequal slices (-ts 0.31,0.27,0.42), exposing an op whose result depends on aliasing or on reading memory the common arena happened to leave defined; the owner's layout has equal dense slices on the XTXs and none on the R9700, and is unaffected. Next: localise (first divergent layer/op) with GPT in a new session.

2026-10-07 COST AND OUTPUT OF 1340 EXPLAINED (build b-metamem-b11402g2 / h2, production row split, ctx 49152, 8K depth, 256 decoded tokens). OUTPUT: GGML_CUDA_DISABLE_FUSION=1 makes the common-arena and per-device-arena runs identical; equal -ts, graphs off, async inputs off do not. Fusion overlap counter (1339): common arena 1020-1032 refusals of 18.6-19.1K checks, per-device 977-985 of 19.7-20.3K - a different set of candidates fuses because ggml_cuda_check_fusion_memory_ranges compares real addresses. So the text difference is kernel selection (fused vs unfused arithmetic), inside the envelope (TV 0.072), not corruption; an identical-output gate cannot be met by any layout change in the row split until FKE01 makes fusion bit-exact or layout-independent. SPEED: not fusion (fewer refusals with the flag). Timer in ggml_backend_meta_alloc_graph: 33-34 calls per request, 3.8 ms per call for three devices, 127-130 ms in total; decode of 256 tokens takes ~3.04 s (common) vs ~3.14-3.19 s (per-device), prefill within 1% - the layout pass accounts for the decode loss (84.0-84.5 vs 80.3-81.6 t/s). THREADING: a version running the three devices' passes on std::thread was written and backed out before any build (owner: threading is dangerous to introduce here). Owner follow-up: threading may be reviewed later if it can be shown safe and it helps, but it must be done carefully - so it is the LAST option, after the non-threaded ones, and only with (1) a written argument for every shared object the passes touch (logical graph read-only, simple-tensor map read-only, per-device gallocr and tensors exclusive, logging), (2) no thread creation per graph (a persistent worker or none), (3) a stress run with identical md5 over many graphs and a sanitizer build if available, (4) owner review of the diff before it is built. NON-THREADED OPTIONS (GPT asked, req_ab35832847a54dbe): skip the pass when the graph shape is unchanged and only re-bind addresses; reserve once for the largest graph so later calls are assign-only; per-device pass only for the device whose plan differs (the zero-attention-share card), others stay on the common arena; fold address binding into init_tensor_impl; replace the per-tensor std::map lookup.

2026-10-07 NON-THREADED REDUCTION MEASURED (GPT commits 5399f67e persistent per-device node lists + split timer, 870e0092 FINAL guarded fast bind; one compile fix on top: the edit that un-statics ggml_gallocr_needs_realloc had a guard matching the pristine text and was skipped - 'static declaration follows non-static' - distinct guard + test added; build b-metamem-b11402k2; production row split, ctx 49152, 8K depth, 256 decoded tokens). Fast bind on: prefill 1054.3/1026.7 vs 1046.9/1021.6, decode 81.4/78.3 vs 84.3/83.9. Fast bind off: prefill 1038.1/1057.4 vs 1058.7/1075.9, decode 80.3/81.4 vs 84.3/84.0. No improvement: decode is still 3-7% below the common arena. TIMER (per request): calls 34, of which a reserve fired in 27; total 120-132 ms, 3.5-4.0 ms per call; split: traversal + map 23 ms, needs_realloc + reserve 62-74 ms, bind 34 ms; graph sizes seen: 7204 nodes x30, 7205 nodes x4. READING: the fast bind almost never applies - 27 of 34 calls re-plan, because the non-reusable graphs (MTP verify batches) differ in size from call to call and a re-plan sizes the per-device plan to the CURRENT graph, so the next larger one re-plans again. The single-traversal change did cut the lookups (23 ms for all devices) but reserve dominates. The same churn must already exist once in the scheduler's own allocator; 1340 adds three more passes of it. MEMORY at ctx 245760 (unchanged by these commits): arena 880.8 / 880.8 / 618.8 MiB with the flag (1341 on, as now in the profile) against 1020.9 x3; indexer cache 1440 / 1440 / 0. Output: owner's layout identical with the flag; production row split different text (fusion selection, FKE01), same md5 as earlier flag-on runs. STATUS: 1340 is correct and saves 140 MiB per XTX and 402 MiB on the R9700 at the full context, but costs 3-7% decode in production. NOT PROMOTED; it stays an opt-in experiment (state untested). Options if it is to be made free, none started: (1) a plan per graph shape that only ever grows (keep the largest size seen per node position instead of re-planning to the current graph), so re-plans stop after the largest verify batch has been seen; (2) derive the per-device plan from the scheduler's plan instead of running an allocator per device; (3) threads, only under the owner's review conditions recorded above. The bigger memory prize on the R9700 (1440 MiB) is already banked by 1341 without any of this.

## Change Log

- 2026-10-06T11:48:08.966158+00:00 (created-by): Created by agent
- 2026-10-06T12:36:51.392852+00:00 (updated-by): Updated: section:notes

## Ledger-events

- chg_20261006_144832_diagnostics-and-experimental-p_1077
- 2026-10-06T14:48:39.466747+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-06T14:48:49.761024+00:00 (updated-by): Updated: section:notes
- 2026-10-06T15:49:27.719538+00:00 (updated-by): Updated: section:notes
- 2026-10-06T17:47:31.424460+00:00 (updated-by): Updated: section:notes


2026-10-07 RESERVE-TIME ARENA REDESIGN: replace 1340's lazy compute-time growth with load-time planning from llama's scheduler reserve graph.

1. **Reserve seam (answer 1): use a scheduler-to-Meta plan-only hook, not a saved graph.** At b11402, `ggml_backend_sched_reserve()` in `ggml/src/ggml-backend.cpp` calls `ggml_backend_sched_split_graph()`, then `ggml_gallocr_reserve_n()`, and only then resets the scheduler. That interval has the exact worst-case scheduler graph and backend assignment alive. 1340 will materialise the logical Meta tensors once, call `ggml_backend_meta_reserve_graph(meta_backend, &sched->graph)`, translate the graph to every simple backend exactly as the existing Meta mapping does, reserve device plans, execute nothing, rotate the temporary Meta tensor containers, then reset. Saving a copy until first compute is rejected: graph/tensor lifetime is harder, allocation would still happen after load, and addresses/fusion choice would still change late.

2. **One physical arena per device (answer 2): plans are metadata only.** Current 1340 `backend_config::arena_plans` stores one `ggml_gallocr` per `(n_nodes,n_leafs)`; `ggml_gallocr_reserve()` gives every plan its own `vbuffer`, so two retained shapes add their buffers (the measured 1420 MiB case). Replace that with one grow-only owner `arena_galloc` per simple backend plus shape `arena_plan_t` gallocr objects reserved with `ggml_gallocr_reserve_n_size(...)` (allocation layout only, no buffer). The owner reserves/grows during scheduler reserve only; compute binds a selected shape plan into the owner's single physical `vbuffer`. `ggml/src/ggml-alloc.c::ggml_gallocr_reserve_n_impl()` is the ownership boundary: retained plan allocators must never own physical buffers.

3. **Sizing source (answer 3): the translated reserve graph is authoritative.** Stock b11402 sizes the scheduler arena from `ggml_backend_sched_reserve()` / `ggml_gallocr_reserve_n()`; that buffer then remains fixed across 2K -> 98K/245K fill, proving llama's measure graph carries the worst-case context-dependent dimensions rather than current `n_kv`. 1340 must translate that same reserve graph, including subset-inactive/axis-split shapes, so KQ masks and gathered rows keep the reserve-time worst-case sizes on devices that own them while inactive devices omit them. `ggml_backend_meta_graph_compute()` is no longer allowed to size/grow an arena in the normal path.

Implementation invariant: after load, normal `ggml_backend_sched_alloc_splits()` may only select a retained shape plan and bind it to the already-reserved per-device arena. A later non-fitting graph is an invariant violation handled by the explicit loud fallback/counter path, not by ordinary growth.
