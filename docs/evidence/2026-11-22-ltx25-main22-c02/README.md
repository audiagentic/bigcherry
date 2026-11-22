# LTX-2.5 IQ8R C02 architecture declarations on released Radiance 2.2

**Date:** 2026-11-22. **Integration branch:** `agent/ltx25-tp3-main22`.
**Base:** `origin/main` at `b80f37d3e` (Radiance 2.2.0 migration #1167 and release 1170.37.0).
**Frozen upstream pin:** `b6b8751cccb0e834a13ae9e5afca824ee9a487aa`; BigCherry patched ABI44.

The LTX foundation and original C01 native builder receipts were cherry-picked onto a **fresh main worktree**, not merged with the older divergent, in-progress bump history. This preserves released serving and source paths without carrying conflicting unreleased changes.

## Measured host-only proof

- The diagnostic `rad0060_ltx2_tp3_linear_declare` patch applies six anchored edits to an **isolated source copy**, is idempotent and leaves the LTX `world_size > 1` refusal intact.
- The actual patched `arch/ltx2/ltx2.cpp` compiled against Radiance 2.2 / ABI44 using GCC 14.
- The C01 test used the actual native `RadBuilder` and read-only `RadFile` model and passed **2,592 fused-row**, **432 biasless output-column** receipts and identified **336 biased outputs** requiring a reduction/bias owner.
- The new C02 test in `tools/lab/radiance/ltx-tp3/native-c01-gate/ltx_c02_lin_declare_test.cpp` includes the **real patched LTX2 architecture implementation** (not a test rewrite of `Lin`). It runs `ltx2::Lin::declare` against `rad_container_sources` and native `Registry` using the ABI-matched reference kernel.
- **3,024 positive C02 declarations:** every DiT block, three uneven 8/8/16, 12/12/8 and 10/10/12 splits, all three ranks, six fused QKV/KV projection types and biasless video FFN-down. Checks cover IQ8R codes, U8/BF16 two-level scale views, BF16 bias part spans, the native GEMM and bias-add op handles, host kernel resolution, and `Builder::finish()` without starting instances.
- **Three fail-closed C02 cases:** a mismatched encoding scale group, an audio FFN-down output with bias, and an unaligned 10/10/12 audio-attention K cut. All refuse before model-weight declarations.
- **Two native source-bound CTests: 2/2 passed.** Both use the actual RAD2 model metadata and kernel tables. The separate synthetic LTX component suite remains 11 CTests.
- The CMake link gate refuses mismatched Radiance versions or source-tree identities; the reference kernel is staged **alone** under the test build directory to prevent unrelated helper libraries in the full SDK kernel directory from loading.

The source file and the read-only local model are **not copied into Git**. No model payload bytes are moved to a GPU and no video-generation work is launched.

## Reproduce (CPU only)

Set `LTX_SOURCE_ROOT` to the exact upstream 2.2 source used to build `LTX_NATIVE_CORE_LIB`; point `LTX_REFERENCE_LIB` to that SDK's `libref.so`; set `BC_LTX_IQ8R_MODEL` explicitly to the local RAD2 container. Use an empty, separate source staging directory.

```sh
python3 tools/tests/lab/test_ltx_tp3_builder_patch.py "$LTX_SOURCE_ROOT" "$LTX_STAGE_ROOT"
cmake -S tools/lab/radiance/ltx-tp3/native-c01-gate \
  -B "${BC_WORK_ROOT}/ltx25-c02-native" \
  -DLTX_SOURCE_ROOT="$LTX_SOURCE_ROOT" \
  -DLTX_STAGE_ROOT="$LTX_STAGE_ROOT" \
  -DLTX_NATIVE_CORE_LIB="$LTX_NATIVE_CORE_LIB" \
  -DLTX_REFERENCE_LIB="$LTX_REFERENCE_LIB" \
  -DLTX_IQ8R_MODEL="$BC_LTX_IQ8R_MODEL"
cmake --build "${BC_WORK_ROOT}/ltx25-c02-native" --parallel 2
cmake --build "${BC_WORK_ROOT}/ltx25-c02-native" --target test
```

Both tests are opt-in, source/build identity checked and leave production recipes unchanged. An absent `LTX_REFERENCE_LIB` disables C02, rather than inventing a kernel.

## What is **not** proven

The C02 test exercises isolated `Lin::declare` calls and finalized **host** op graphs, not the complete architecture `rad_arch_declare` for TP3, full model payload loading, accelerated HIP kernel resolution or execution, distributed Q/K normalization, barriers and all-reduces, video/audio guidance, rank ownership of VAEs, actual image/audio output or a speedup. It does not remove the production single-rank admission gate.

The next implementation is a controlled real LTX model **whole attention block** with global all-head Q/K RMS, correct per-rank activation storage and one bias application after the reduction; GPU qualification must first obtain an appropriate device lease and compare actual numerical output with TP1.
