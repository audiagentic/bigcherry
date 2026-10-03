#!/bin/bash
# One decode-mode rocprof pass (~10K cached) with extra env, then the per-GPU op-class census (rank-census.py).
# Usage: census-run.sh <llama-server> <out-dir> [env assignments...]
set -u
bin=$1 out=$2
shift 2
here=$(cd "$(dirname "$0")" && pwd)
env "$@" DEPTH=${DEPTH:-8192} DECODE_N=256 bash "$here/long-ctx-profile.sh" "$bin" "$out" decode 2>&1 | grep -E "^decode window|^timing"
python3 "$here/rank-census.py" "$out"
