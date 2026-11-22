# LTX-2.5 IQ8R: diagnostic deferred-bias output-column declarations

**Date:** 2026-11-22. **Branch:** `agent/ltx25-tp3-main22`.
**Baseline:** released Radiance 2.2.0, ABI44; continuation of PR #1262.

## Result and limits

The opt-in `rad0060_ltx2_tp3_linear_declare` source patch now admits
a **biased** column-sharded IQ8R projection in `Lin::declare` *only when*
the caller selects `C01Slice::columns_deferred` and supplies exactly one
bias checkpoint source. It deliberately declares **no per-rank bias
weight or `table_add` op** for that partial GEMM. `Lin::deferred_bias`
records that this output is **not final** and requires one full-width
bias application **after** a future inter-rank reduction. Neither the
reduction nor its unique owner has been implemented. A biasless
`C01Slice::columns` continues to take zero bias sources. The original
TP1 path is unchanged; LTX production still refuses world sizes >1.

Source-exact `RadBuilder` tests read the real LTX-2.5 IQ8R container,
include the patched actual LTX `Lin::declare`, finish the host graph,
and verify IQ8R weight/scale views and native host op resolution.
Across all 48 DiT blocks, three proposed head splits and three ranks:

- **3,024** existing fused-QKV/KV and biasless video FFN-down declarations passed.
- **2,640** additional biased output-column declarations passed with
  `deferred_bias=true`, no rank-local bias weight and no bias-add op.
  Tested output types: video/audio self and cross attention, both
  cross-modal directions and audio FFN-down.
- **384** rank-local output K cuts violate an IQ8R group boundary.
  Every model shape was checked; the native fail-closed branch was
  invoked for each distinct invalid case in block 0 to avoid duplicate
  expected diagnostics across 48 identical block layouts.
  The four audio-side output types in the 10/10/12 split fail on
  ranks 0 and 1; the rank-2 cut is group-aligned.
- Four explicit negative native checks: incompatible source scale
  group; biased output without deferral; invalid audio K cut;
  deferral without a bias source.
- The full patched `arch/ltx2/ltx2.cpp` compiled against the exact
  ABI44 headers. Patch applicability and idempotence: **7 edits**.

## Proof boundaries

This is a **host-only diagnostic declaration**. A program cannot use
deferred bias as a final projection without a correct collective and
a unique bias owner. Model payload loading, the complete distributed
DiT graph, all-head Q/K RMS, runtime collectives, GPU kernels, output
quality and video speed remain **unproven**.

Only the unselected diagnostic patch is modified. No production source
recipe or TP>1 admission is changed, and no GPU resource is used.

## Reproduction

Stage a clean, pinned Radiance 2.2 LTX source into an *empty* work-root
directory using `tools/tests/lab/test_ltx_tp3_builder_patch.py`.
Build the opt-in `native-c01-gate` with the source-aligned 2.2 native
core, its isolated reference `libref.so`, and a read-only model path.

```sh
python3 tools/tests/lab/test_ltx_tp3_builder_patch.py "$LTX_SOURCE_ROOT" "$LTX_STAGE_ROOT"
cmake -S tools/lab/radiance/ltx-tp3/native-c01-gate \
  -B "${BC_WORK_ROOT}/ltx25-c03-native" \
  -DLTX_SOURCE_ROOT="$LTX_SOURCE_ROOT" \
  -DLTX_STAGE_ROOT="$LTX_STAGE_ROOT" \
  -DLTX_NATIVE_CORE_LIB="$LTX_NATIVE_CORE_LIB" \
  -DLTX_REFERENCE_LIB="$LTX_REFERENCE_LIB" \
  -DLTX_IQ8R_MODEL="$BC_LTX_IQ8R_MODEL"
cmake --build "${BC_WORK_ROOT}/ltx25-c03-native" --parallel 2
cmake --build "${BC_WORK_ROOT}/ltx25-c03-native" --target test
```

The host-only component tests are independent of the model and kernel
installation. They intentionally remain CI-safe.
