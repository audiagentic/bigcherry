#!/bin/bash
# 4 independent contract-campaign sessions for patch 0840 adaptive AllReduce (PGC09; 0860+1225 as --common-patches) on Brutus dual gfx1100.
# 5-minute cooldown between sessions (PA35: back-to-back sessions without a gap cause clock instability).
set -u
prefix=${1:?usage: run-pgc09-sessions.sh <run-name-prefix, e.g. t-0840-gfx1100>}
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/vault/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
M=/mnt/vault/llm-models
for s in 1 2 3 4; do
  jobs=$(mktemp)
  echo "VIS=0,1 MODEL=$M/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf 0840_hybrid_allreduce_dispatch 0840_hybrid_allreduce_dispatch/pgc09 gfx1100 0,1 $prefix-s$s --common-patches 0860_allreduce_provider_cli,1225_hi85_nccl_heterogeneous_arch_guard --producer-corpus tools/bigcherry/bench/corpora/mtp-27b-v1.jsonl" > "$jobs"
  bash tools/lab/plan-qualification/queue.sh "$jobs"
  echo "SESSION_${s}_EXIT=$? $(date -Is)"
  rm -f "$jobs"
  [ "$s" -lt 4 ] && sleep 300
done
echo ALL_SESSIONS_DONE
