#!/bin/bash
# QFP18 promotion GPU session (runs after queue-peak):
# (1) native llama.cpp (llama-native source, no patches, no --allreduce) vs v6 at 64K ctx f16, depths 8K + 48K,
#     ABA with v6 as the A arm (BIGCHERRY_FEATURES=flashnext) and native as the middle arm, same flags otherwise;
# (1b) FMTP03 screen: v6 vs v6 + BIGCHERRY_MTP_AHEAD=1 (same binary) at 24K and 80K, 240K f16;
# (2) cross-model no-regression: Qwen3.8-27B dual-XTX production config, promoted base (stock-none on the bigcherry
#     source) vs base + the Flash-Next patch set (deploy-v5-plus-1327) with every flag at its default, ABBA, 2 depths.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=65536 TS=0.31,0.27,0.42 B=512 DECODE_N=256
export EXTRA_OT='^token_embd\.weight$=CPU'
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
R=/mnt/data/bigcherry-work/runs
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-native llama-native:stock:linux-multi stock-none gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 BUILD b-base bigcherry:stock:linux-multi stock-none gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 BUILD b-v6 bigcherry:stock:linux-multi deploy-v5-plus-1327 gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT nat-d8k tools/lab/flash-next/native-ab.sh 8192 @b-v6 @b-native $R/flashnext-native-d8k
VIS=0,1,2,3 SCRIPT nat-d48k tools/lab/flash-next/native-ab.sh 49152 @b-v6 @b-native $R/flashnext-native-d48k
VIS=0,1,2,3 BUILD b-ahead bigcherry:stock:linux-multi deploy-v6-plus-ahead gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT ahead-d24k tools/lab/flash-next/ahead-ab.sh 24576 @b-ahead $R/flashnext-ahead-d24k
VIS=0,1,2,3 SCRIPT ahead-d80k tools/lab/flash-next/ahead-ab.sh 81920 @b-ahead $R/flashnext-ahead-d80k
VIS=0,1 SCRIPT p27b tools/lab/flash-next/prod27b-ab.sh @b-base @b-v6 $R/prod27b-promote
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
for d in d8k d48k; do echo "== native $d"; grep -E "^base-|^new|SERVER_FAILED" $R/nat-$d.log; done
for d in d24k d80k; do echo "== ahead $d"; grep -E "^base-|^new|SERVER_FAILED" $R/ahead-$d.log; md5sum $R/flashnext-ahead-$d/*/*.greedy.txt | awk '{print $1}' | sort | uniq -c; grep -h BIGCHERRY_MTP_AHEAD $R/flashnext-ahead-$d/new/*.log | tail -1; done
echo "== 27B"; grep -E "^d[0-9]|SERVER_FAILED|^ +[0-9]" $R/p27b.log
echo ALL_JOBS_DONE
