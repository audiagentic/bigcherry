# 1002_hip_unsafe_math_opt_in: Upstream backport: make -funsafe-math-optimizations opt-in for HIP builds

**Status:** untested
**Group:** upstream-fixes
**Plan item:** none

## What it does

Adds GGML_HIP_UNSAFE_MATH (default OFF) gating -funsafe-math-optimizations in ggml/src/ggml-hip/CMakeLists.txt, which this project's tree previously enabled unconditionally for every HIP build.

## Why

Unsafe math reassociates floating-point reductions, which can flip a greedy argmax and break byte-identical output for MTP speculative decoding at temperature 0. Verified directly on hardware: gfx1201 was byte-identical, but gfx1100 diverged specifically on speculative-vs-non-speculative runs, confirming the mechanism on a third architecture upstream never tested.

## Upstream / provenance

Cherry-picked from open upstream PR https://github.com/ggml-org/llama.cpp/pull/26696. Upstream later merged its own version (e79e4bf6, by deletion rather than an opt-in gate); this backport's anchor was adapted to the surviving GGML_HIP_EXPORT_METRICS block so its own OFF-by-default escape hatch is preserved.
