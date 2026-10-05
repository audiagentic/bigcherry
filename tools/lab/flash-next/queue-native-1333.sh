#!/bin/bash
# QFP22: native llama.cpp comparison for 1333 - native (source llama-native, no patches) vs native + 1333 alone,
# Qwen3.8-27B dual-XTX production config (prod27b-ab.sh: ABBA at 10K and 32K, prefill/decode/acceptance/greedy md5).
# Build run names carry the pin so a later pin gets fresh builds.
# Usage: queue-native-1333.sh <pin tag, e.g. b11402>
set -u
PIN=${1:?pin tag for the build run names}
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
R=/mnt/data/bigcherry-work/runs
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-native-$PIN llama-native:stock:linux-multi stock-none gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 BUILD b-native1333-$PIN llama-native:stock:linux-multi native-plus-1333 gfx1100,gfx1201,gfx1030
VIS=0,1 SCRIPT native1333-$PIN-27b tools/lab/flash-next/prod27b-ab.sh @b-native-$PIN @b-native1333-$PIN $R/native1333-$PIN-27b
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo "== 27B (A = native $PIN, B = native + 1333)"; grep -E "^d[0-9]|SERVER_FAILED|^ +[0-9]" $R/native1333-$PIN-27b.log
echo ALL_JOBS_DONE
