#!/bin/bash
# QFP18 promotion GPU session (runs after queue-peak):
# (1) native llama.cpp (llama-native source, no patches, no --allreduce) vs v6 at 64K ctx f16, depths 8K + 48K,
#     ABA with v6 as the A arm (BIGCHERRY_FEATURES=flashnext-v6) and native as the middle arm, same flags otherwise;
# (2) cross-model no-regression: Qwen3.8-27B dual-XTX production config, promoted base (stock-none on the bigcherry
#     source) vs base + the Flash-Next patch set (deploy-v5-plus-1327) with every flag at its default, ABBA, 2 depths.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=65536 TS=0.31,0.27,0.42 B=512 DECODE_N=256
export EXTRA_OT='^token_embd\.weight$=CPU'
until grep -q ALL_JOBS_DONE /mnt/data/bigcherry-work/runs/queue-peak.log 2>/dev/null; do sleep 20; done
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
R=/mnt/data/bigcherry-work/runs
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-native llama-native:stock:linux-multi stock-none gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 BUILD b-base bigcherry:stock:linux-multi stock-none gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 BUILD b-v6 bigcherry:stock:linux-multi deploy-v5-plus-1327 gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT nat-d8k tools/lab/flash-next/native-ab.sh 8192 @b-v6 @b-native $R/flashnext-native-d8k
VIS=0,1,2,3 SCRIPT nat-d48k tools/lab/flash-next/native-ab.sh 49152 @b-v6 @b-native $R/flashnext-native-d48k
VIS=0,1 SCRIPT p27b tools/lab/flash-next/prod27b-ab.sh @b-base @b-v6 $R/prod27b-promote
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
for d in d8k d48k; do echo "== native $d"; grep -E "^base-|^new|SERVER_FAILED" $R/nat-$d.log; done
echo "== 27B"; grep -E "^d[0-9]|SERVER_FAILED|^ +[0-9]" $R/p27b.log
echo ALL_JOBS_DONE
