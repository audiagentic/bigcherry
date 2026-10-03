# 0110_campaign_tune_record_build: CMake options for HIP tune/record campaign build (HI02, PA27)

**Status:** validated
**Plan item:** PA27

## What it does

Adds GGML_HIP_AUTOTUNE, GGML_HIP_AUTOTUNE_RECORD, GGML_HIP_DISPATCH_DIAGNOSTICS,
GGML_HIP_WORKSPACE_METRICS, and GGML_HIP_ROUTING_TRANSFORM to
ggml/CMakeLists.txt (with configure-time validation) and turns them into
HIP-backend compile definitions plus the tuner/record/diagnostics source list
in ggml/src/ggml-hip/CMakeLists.txt.

HI168: the coverage implementation is linked only for diagnostic, record, or
tune builds; a pure replay build never carries its counter storage and
reporting code (that exclusion lives in `0100_cmake_options`).

Split out of `0100_cmake_options` by PA27 (dev-gpt-agent design,
req_42717536998244ea) so a pure replay/serving build never carries any
campaign-only code. Not declared as `requires = ["0100_cmake_options"]` --
PA28's serving-core/campaign-support composition needs to add this package
to sets that also carry `0100` independently. Instead, every campaign-only
branch fails closed at CMake configure time if `0100`'s
`_BC_HIP_SERVING_BUILD_PLUMBING` marker is absent.

## Why

Tuning/record builds need their own build switches, and illegal build
combinations must fail at configure time rather than produce a silently
incomplete or inert build: GGML_HIP_AUTOTUNE=ON with GGML_HIP=OFF, dispatch
combined with GGML_CUDA_FORCE_MMQ/GGML_CUDA_FORCE_CUBLAS (which would hide
candidate families from measurement), GGML_HIP_ROUTING_TRANSFORM without both
GGML_HIP_AUTOTUNE and GGML_HIP_AUTOTUNE_RECORD, and any campaign-only option
activated without the serving package present.

## Upstream / provenance

Local design, part of this project's own HIP measured-dispatch framework (HI02).
