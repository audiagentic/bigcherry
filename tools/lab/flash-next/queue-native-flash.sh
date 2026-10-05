#!/bin/bash
# Native llama.cpp baseline for Flash-Next on the current pin: our production profile build (A arms,
# BIGCHERRY_FEATURES=flashnext) vs native llama.cpp (middle arm, no patches), 64K ctx f16 (native cannot load the
# 240K f16 deployment), depths 8K and 48K, native-ab.sh ABA. Uses existing build runs; builds nothing.
# Usage: queue-native-flash.sh <our build run> <native build run> <tag> [wait=<log with ALL_JOBS_DONE>]
set -u
OURS=${1:?our build run, e.g. b-mixauto}
NATIVE=${2:?native build run, e.g. b-native-b11402}
TAG=${3:?tag}
shift 3
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=/mnt/data/bigcherry-work/runs
for a in "$@"; do case "$a" in wait=*) until grep -q "^ALL_JOBS_DONE" "${a#wait=}" 2>/dev/null; do sleep 20; done ;; esac; done
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=65536 TS=0.31,0.27,0.42 B=512 DECODE_N=256
export EXTRA_OT='^token_embd\.weight$=CPU'
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
cat > "$jobs" <<JOBS
VIS=0,1,2,3 SCRIPT $TAG-d8k tools/lab/flash-next/native-ab.sh 8192 @$OURS @$NATIVE $R/$TAG-d8k
VIS=0,1,2,3 SCRIPT $TAG-d48k tools/lab/flash-next/native-ab.sh 49152 @$OURS @$NATIVE $R/$TAG-d48k
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
for d in d8k d48k; do echo "== $d (base = $OURS, new = native $NATIVE)"; grep -E "^base-|^new|SERVER_FAILED" $R/$TAG-$d.log; done
echo ALL_JOBS_DONE
