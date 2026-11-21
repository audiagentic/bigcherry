# LTX-2.5 IQ8R C01 on Radiance 2.2 — native builder receipts

Date: 2026-11-21. Working branch: `agent/ltx25-tp3-2p2`.
Pinned candidate: Radiance 2.2.0 `b6b8751cccb0e834a13ae9e5afca824ee9a487aa`, patched ABI44. The LTX foundation was merged into an isolated worktree from the 2.2 bump branch; production source recipes and the LTX `world_size > 1` admission refusal remain unchanged.

## Proof

The diagnostic `rad0060_ltx2_tp3_linear_declare` patch applied and reapplied idempotently (six edits), then the entire patched `arch/ltx2/ltx2.cpp` compiled against the 2.2 ABI44 C++ SDK with GCC 14/C++23.

The new opt-in `tools/lab/radiance/ltx-tp3/native-c01-gate/ltx_c01_weight_receipt_test.cpp` uses the **real Radiance native 2.2 core**, its `RadFile` reader and `RadBuilder` / `LinearFP8::span_parts` APIs, not mocks or a duplicated model parser. For all DiT blocks in the real IQ8R RAD2 container and 8/8/16, 12/12/8, 10/10/12 head splits, it checked:

- 2,592 native builder fused QKV/KV row-shard receipts (I8 codes; U8+BF16 two-level scale views; BF16 bias part spans)
- 432 native builder biasless video FFN-down column-shard receipts with codes and two-level scale views
- 336 biased output tensors marked as requiring **one bias application after the inter-rank reduction**, not admitted by C01
- 10/10/12 invalid audio-attention K cut refused by C01's scale-group geometry

Output: `LTX2 native ABI44 IQ8R Builder receipts: rows=2592 cols=432 blocked=336`.
This is **native weight declaration** evidence; it does not run `ltx2::Lin::declare` as part of a complete model graph, finish/resolve HIP kernels, load model weight buffers to VRAM, execute TP3 attention or denoise frames.

Independent real-model Python raw-plane oracle: 1,306 IQ8R tensors structurally valid; 1,920 raw code/scale column-shard roundtrips; 384 invalid boundary-row splits (the four incompatible audio attention outputs ×48 blocks ×2 boundary rows).

`C01Slice::valid` now consumes the encoding-derived scale group; the native C01 preflight checks `RAD_I8_GROUP_WIDE`, rather than hand-written group-width assumptions. The native CMake link test refuses a mismatched core: a deliberately older Radiance 2.1 archive produced `C01 cannot link a different Radiance version or source build`.

## Reproduce (read-only model, no GPU)

Use the active project's original frozen 2.2 source/build and a **separate empty stage** for the anchored C01 patch. Never patch an upstream source in place. Then:

```sh
python3 tools/tests/lab/test_ltx_tp3_builder_patch.py "$LTX_SOURCE_ROOT" "$LTX_STAGE_ROOT"
cmake -S tools/lab/radiance/ltx-tp3/native-c01-gate \
  -B "${BC_WORK_ROOT}/ltx-c01-build" \
  -DLTX_SOURCE_ROOT="$LTX_SOURCE_ROOT" \
  -DLTX_STAGE_ROOT="$LTX_STAGE_ROOT" \
  -DLTX_NATIVE_CORE_LIB="$LTX_NATIVE_CORE_LIB" \
  -DLTX_IQ8R_MODEL="$BC_LTX_IQ8R_MODEL"
cmake --build "${BC_WORK_ROOT}/ltx-c01-build" --parallel 2
cmake --build "${BC_WORK_ROOT}/ltx-c01-build" --target test
```

`LTX_NATIVE_CORE_LIB` points to that exact build's `core/librad_core.a`; CMake confirms build version and original source root before linking the native test. With the core archive omitted, the gate compiles the patched LTX source only. `BC_LTX_IQ8R_MODEL` is an opt-in local path, never committed.

Further CPU-only acceptance: original 11/11 LTX CTests passed; five focused Python modules yielded 7 passed and 5 model-opt-in skips; explicit real-model plane and raw-shard runs passed. Standard core guards and patch registry checks to be recorded separately.

## Outstanding

Actual LTX rank-local buffers and `Lin::declare` graph/weight loader admission, global all-head Q/K RMS, inter-rank collectives, bias-once ownership, full DiT dual-modality distribution, video/audio output ownership, GPU code objects, quality and real video speed remain **unimplemented/unproven**. Keep the production world-size refusal intact. Future testing must use the GPU lease.
