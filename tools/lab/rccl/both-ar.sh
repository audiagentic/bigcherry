#!/bin/bash
# Run both AllReduce microbenchmarks (RCCL and CPU-root), synchronised-per-call and stream-ordered chains.
set -u
out=$1; mkdir -p "$out"
here=$(cd "$(dirname "$0")" && pwd)
bash "$here/ar-latency.sh" "$out" 2>&1 | grep -E "ranks=|=="
bash "$here/cpu-root-ar.sh" "$out"
