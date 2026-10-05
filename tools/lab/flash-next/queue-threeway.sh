#!/bin/bash
# Three-way baseline on the current pin: native llama.cpp vs BigCherry base (bigcherry source, no enhancement
# patches: experiment stock-none) vs BigCherry patched (production build). Builds only the base binary; native and
# patched come from existing build runs.
#   27B dual-XTX production config (prod27b-ab.sh ABBA, 10K + 32K): native vs base, then base vs patched.
#   Flash-Next 64K ctx f16 (native-ab.sh ABA, 8K + 48K): patched (A arms, flashnext profile) vs base (middle arm).
#   (patched vs native for Flash-Next is queue-native-flash.sh.)
# Usage: queue-threeway.sh <pin tag> <native build run> <patched build run> [wait=<log with ALL_JOBS_DONE>]
set -u
PIN=${1:?pin tag}
NATIVE=${2:?native build run}
PATCHED=${3:?patched build run}
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
BASE=b-base-$PIN
jobs=$(mktemp)
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD $BASE bigcherry:stock:linux-multi stock-none gfx1100,gfx1201,gfx1030
VIS=0,1 SCRIPT 3way-$PIN-27b-native-base tools/lab/flash-next/prod27b-ab.sh @$NATIVE @$BASE $R/3way-$PIN-27b-native-base
VIS=0,1 SCRIPT 3way-$PIN-27b-base-patched tools/lab/flash-next/prod27b-ab.sh @$BASE @$PATCHED $R/3way-$PIN-27b-base-patched
VIS=0,1,2,3 SCRIPT 3way-$PIN-flash-d8k tools/lab/flash-next/native-ab.sh 8192 @$PATCHED @$BASE $R/3way-$PIN-flash-d8k
VIS=0,1,2,3 SCRIPT 3way-$PIN-flash-d48k tools/lab/flash-next/native-ab.sh 49152 @$PATCHED @$BASE $R/3way-$PIN-flash-d48k
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo "== 27B: A = native $NATIVE, B = base $BASE"; grep -E "^d[0-9]|SERVER_FAILED" $R/3way-$PIN-27b-native-base.log
echo "== 27B: A = base $BASE, B = patched $PATCHED"; grep -E "^d[0-9]|SERVER_FAILED" $R/3way-$PIN-27b-base-patched.log
for d in d8k d48k; do echo "== Flash-Next $d: base arms = patched $PATCHED, new = base $BASE"; grep -E "^base-|^new|SERVER_FAILED" $R/3way-$PIN-flash-$d.log; done
echo ALL_JOBS_DONE
