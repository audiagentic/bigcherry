#!/bin/bash
# Execute radiance under ROCprofiler-SDK runtime tracing: HIP runtime, kernels and memory copies.
set -eu
: "${ROCPROF_OUT:?set ROCPROF_OUT}"
mkdir -p "$ROCPROF_OUT"
exec rocprofv3 --runtime-trace --output-format csv --output-directory "$ROCPROF_OUT" -- "$@"
