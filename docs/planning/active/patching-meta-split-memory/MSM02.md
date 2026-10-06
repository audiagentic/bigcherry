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

## Change Log

- 2026-10-06T11:48:08.966158+00:00 (created-by): Created by agent
- 2026-10-06T12:36:51.392852+00:00 (updated-by): Updated: section:notes
