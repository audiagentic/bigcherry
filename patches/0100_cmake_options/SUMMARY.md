# 0100_cmake_options: CMake options for HIP replay/serving build (HI02, PA27)

**Status:** validated
**Plan item:** none

## What it does

Adds GGML_HIP_DISPATCH_REPLAY and related build options to ggml/CMakeLists.txt
(with configure-time validation) and turns them into HIP-backend compile
definitions plus the production dispatch/replay source list in
ggml/src/ggml-hip/CMakeLists.txt. Sets the `_BC_HIP_SERVING_BUILD_PLUMBING`
non-cache marker that `0110_campaign_tune_record_build` reads to fail closed
if its own options are activated without this package applied.

PA27 narrowed this package to only what a replay/serving build needs; the
tune/record-only options, tuner source, and campaign-only diagnostics moved
to `0110_campaign_tune_record_build`. There is no SQLite link edit anywhere
in the split (the declaration-only, no-consumer GGML_HIP_AUTOTUNE_SQLITE
option this package used to carry was dead code and was removed).

## Why

Measured dispatch needs its own build switches, and an illegal build
combination must fail at configure time rather than produce a silently
incomplete or inert build: GGML_HIP_DISPATCH_REPLAY=ON with GGML_HIP=OFF, and
dispatch combined with GGML_CUDA_FORCE_MMQ/GGML_CUDA_FORCE_CUBLAS (which
would hide candidate families from measurement).

## Upstream / provenance

Local design, part of this project's own HIP measured-dispatch framework (HI02).
