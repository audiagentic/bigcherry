# 1232_hi81_windows_cxx_hipcc_flags_reach_compile: Make CMAKE_HIP_FLAGS-gated options actually reach the compiler on Windows Ninja+Clang HIP builds (HI81)

**Status:** untested
**Group:** core
**Plan item:** HI81

## What it does

Under the CXX_IS_HIPCC branch (always taken on Windows, since CMake's HIP language isn't used there), appends the same flags to the ROCm source files' real CXX COMPILE_FLAGS via set_property(APPEND_STRING), composing with the pre-existing Windows-Debug workaround already at that spot.

## Why

On Windows, ROCm source files are compiled as plain CXX rather than through CMake's HIP language rule, so anything appended to CMAKE_HIP_FLAGS (including this project's own GGML_HIP_UNSAFE_MATH option, patch 1002) was a complete no-op there -- confirmed by comparing CMakeCache.txt against ninja -t commands output.

## Upstream / provenance

Local design, a real build-configuration bug fix found by reading the CMake source directly (HI81).
