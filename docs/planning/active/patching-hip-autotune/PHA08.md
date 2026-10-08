---
id: PHA08
order: 8
plan: patching-hip-autotune
state: pending
created-at: '2026-09-11T22:57:38.584940+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: L
---

# HIP Flash-Attention D=72 VLM aperture violation: root-cause before fallback (#28608/#28664)

## Description and 2026-10-08 disposition

**Current baseline: llama.cpp b11474, commit b9acf138.** The previous b11402 and 050439614 baselines are historical, not current. Issue #28608 remains open (updated 2026-09-10); PR #28664 closed unmerged. Current upstream master and pinned b11474 have byte-identical ggml/src/ggml-cuda/fattn-tile.cuh; neither pinned nor current tools/mtmd/clip.cpp has the proposed D=72 HIP fallback. There is no established BigCherry b11474 reproduction, fix, or measured improvement.

**Do not run the previously proposed generic D=72 partial-vector tail-guard experiment first.** HIP ggml_cuda_get_max_cpy_bytes() returns 16 B (common.cuh:399-408). In fattn-tile.cuh, contiguous D=72 Q F32 loads use 4 F32/vector (72/4=18), K/V half2 copies use 4 half2/vector (36/4=9), and output F32 stores use 2 float2/vector (36/2=18). The respective 288 B Q/output and 144 B K/V rows are 16 B-aligned in size. Eighteen host-only vector/row arithmetic fixtures (six column geometries times three paths) passed. **This rules out a partial-vector tail in these visible contiguous paths, not all out-of-bounds accesses or unaligned runtime views.** Do not claim the kernel is safe from this static check. BCOP25's proposed tail-guard action is superseded by BCOP66.

## Exact implementation map

- tools/mtmd/clip.cpp::clip_graph::build_attn (~750): when CLIP_FLASH_ATTN_TYPE_ENABLED, Q/K/V permute and K/V F16 cast precede ggml_flash_attn_ext; the existing else arm computes matmul -> softmax -> matmul. This is the *only* permitted emergency fallback boundary. Before a HIP guard, verify GGML_USE_HIP (or equivalent backend identity) is actually visible in mtmd compilation; do not assume the backend's compile define propagates to tools.
- ggml/src/ggml-cuda/fattn-tile.cuh::launch_fattn_tile_switch_ncols1 (~1149): on HIP and DKQ<=128, Q->ne[1]>32/ncols2 selects 64 columns; ncols2=1 yields the reported flash_attn_tile<72,72,64,1,false>. The 32-column override made upstream crashes earlier, not safer. ggml_cuda_fattn_tile_get_config_amd_rdna (~237-257) has D=72 configurations for 2/4/8/16/32/64 columns, nbatch_K=72.
- Read/write anchors: flash_attn_tile_load_tile (~378/428), flash_attn_tile_iter_KQ (~485), flash_attn_tile_iter (~560), flash_attn_tile (~793), Q load (~905-950), K/V iteration (~953-979), output store (~1082-1128). Inspect actual Q/K/V/mask/KV_max/dst extents, nb strides, GQA head/sequence indices, k_VKQ_max vs ne11, DVp=128 scratch padding, grid dimensions and graph/stream lifetime. Static vector divisibility does not settle any of these.
- ggml/src/ggml-cuda/common.cuh::ggml_cuda_get_max_cpy_bytes (~399): HIP uses 16-byte copies. No new allocator, dispatch registry, duplicate tile kernel or scheduler is permitted. Existing HIP autotune owns architecture/shape policy; PHA08 owns only this CLIP D=72 safety boundary.

## Cheapest next discriminator and algorithm

1. Capture the actual failing graph and runtime Q/K/V/mask/dst ne, nb, dtype, device, base allocation/size, pointer offsets, ncols1/ncols2, nbatch_fa/nbatch_K, grid/block, KV_max and mask shape, clip image token count, ROCm/driver/compiler and mmproj/model hashes.
2. Build a host-side address replay for the exact launch: for every logical vector read/write, calculate base + sequence/head/row/column strides + vector bytes; compare against the owning allocation, not just tensor ne. Include padded last KV batch, zero/nonzero mask, ncols2=1/2/4/8, noncontiguous views, and D=64/72/80 controls. A host replay is a discriminator, not GPU memory-safety qualification.
3. If host replay fails, localize first invalid Q/K/V/mask/KV_max/output address; implement the smallest helper/stride fix and validate against the stock graph. If replay passes but hardware faults, isolate stream/lifetime and scratch/launch; instrument asynchronous ownership and synchronize at the CLIP graph boundary. Inspect gfx1100 ISA only for the narrowed candidate. Revisit a tail guard only if actual runtime layout or instruction width disproves the 16-byte source arithmetic; otherwise do not build one.
4. If no bounded kernel fix is proven, a HIP + CLIP + D=72 fallback to the existing matmul branch is the sole allowed workaround, after confirming backend guard compilation and selection. Never disable main-model FA, all HIP FA, other CLIP dimensions, or NVIDIA/Vulkan. Preserve non-HIP fallback semantics and remove disposable diagnostics.
5. Upstream #29435 / d89651a7 is NVIDIA DGX Spark scheduling only, not a D=72 correctness repair. vLLM ROCm's supported-head-size gating with a separate Triton fallback is a *policy example*, not a drop-in llama.cpp kernel or AMD performance measurement. Upstream issue #28608's reporter also noted a gfx1030 aperture symptom without vision at 262144 context; do not conflate unrelated faults.

## Qualification and terminal gates

Baseline B = b11474 stock. Controls = B with CLIP matmul attention, D=64/80 healthy CLIP FA, and text-only generation. Subjects = instrumented B, a proven minimal fix, or a narrowly gated fallback. Reproduce on 1x and 2x gfx1100 XTX with 1024/1600/2048/2560px Qwen/Gemma mmproj; compare gfx1201 where same kernel dispatches and gfx1030 as separately labelled control. Test ncols 32/64 diagnostically only. Keep long-context/MTP and main-model attention results separate from CLIP encoding.

For any safety claim: >=20 repeated 2048/2560px encodes on the failing lane, zero HSA aperture faults and zero verified invalid accesses; parity against CLIP matmul embeddings/logits/greedy output within registered tolerance, graph capture/replay, multi-image and multi-request same-process, resource reuse and teardown. Record per-kernel and encode latency, TTFT, allocations/peak VRAM, synchronization and bytes transferred, with at least four independent sessions where available. Performance promotion of a kernel fix requires >=5% *measured* VLM encode improvement versus safe fallback and <=2% D=64/80 regression, without text-only regression; include variance/CI. Faster because work disappeared is a failure.

Terminal outcomes: (A) b11474 cannot reproduce across full stress matrix -> close without patch; (B) reproducible and localized -> minimal kernel fix, full correctness/performance gate; (C) reproducible but unlocalized -> retain only a verified narrow CLIP fallback and document upstream blocker; (D) disproven partial-vector hypothesis -> no tail-guard task. **No build, HIP kernel run, new hardware benchmark or safety validation occurred in this 2026-10-08 audit.**

## Sources / ownership

- https://github.com/ggml-org/llama.cpp/issues/28608
- https://github.com/ggml-org/llama.cpp/pull/28664
- https://github.com/ggml-org/llama.cpp/blob/b9acf138/ggml/src/ggml-cuda/fattn-tile.cuh
- https://github.com/ggml-org/llama.cpp/blob/b9acf138/ggml/src/ggml-cuda/common.cuh
- https://github.com/ggml-org/llama.cpp/blob/b9acf138/tools/mtmd/clip.cpp
- https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/rocm_attn.py

PHA08 is authoritative. BCOP25 is superseded; BCOP66 tracks only the remaining bounded qualification. Recent QFP/PA/MTP/MMQ/PRBE active work is protected and untouched.

## Change Log

- 2026-09-11T22:57:38.584940+00:00: Created by agent.
- 2026-10-05: BCOP21 audit backfill added root-cause gate.
- 2026-10-05: Transplanted structured D=72 root-cause/tail-geometry plan from `automation-qfp-indexer-20261004`.
- 2026-10-05: Updated baseline to b11402; bounded the tail-safe diagnostic prototype; classified fresh upstream #29435 as NVIDIA-only scheduling evidence and added explicit stop gates.

## Ledger-events

- chg_20260911_225756_added-six-provenance-rich-buil_2622
- 2026-09-11T22:57:56.719160+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-05T04:55:02.911053+00:00 (updated-by): Updated: section:notes
