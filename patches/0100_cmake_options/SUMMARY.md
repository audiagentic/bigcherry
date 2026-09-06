# 0100_cmake_options: CMake options for HIP measured dispatch (HI02)

**Status:** validated
**Group:** core
**Plan item:** none

## What it does

Adds GGML_HIP_AUTOTUNE and related build options to ggml/CMakeLists.txt (with configure-time validation) and turns them into HIP-backend compile definitions plus SQLite linkage in ggml/src/ggml-hip/CMakeLists.txt.

HI168: the coverage implementation is linked only for diagnostic, record or
tune builds. Production replay excludes its counter storage and reporting code.

## Why

Measured dispatch needs its own build switches, and two illegal build combinations must fail at configure time rather than produce a silently incomplete or inert build: GGML_HIP_AUTOTUNE=ON with GGML_HIP=OFF, and dispatch combined with GGML_CUDA_FORCE_MMQ/GGML_CUDA_FORCE_CUBLAS (which would hide candidate families from measurement).

## Upstream / provenance

Local design, part of this project's own HIP measured-dispatch framework (HI02).
