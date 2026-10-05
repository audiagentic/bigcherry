#!/bin/bash
# Default-on check for the fused Q8_1 decode path (1307 cache + 1309/1310/1311/1312/1313): builds the production
# recipe at the current HEAD (flags unset = on) and compares it with a build from before the default flip.
#   27B dual-XTX production config, NO runtime flags on either arm (prod27b-ab.sh clears them): old defaults (off)
#   vs new defaults (on) - the gain every model without a profile now gets. Then the new build with the path switched
#   off explicitly, to prove the off switches work.
#   Flash-Next 24K MTP decode with the flashnext profile on both: must be unchanged (the profile enabled the path
#   before; now the default does).
# Usage: queue-defaults.sh <tag> <build run from before the flip> [wait=<log with ALL_JOBS_DONE>]
set -u
TAG=${1:?tag for the new build run}
OLD=${2:?build run from before the default flip, e.g. b-mixauto}
shift 2
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=/mnt/data/bigcherry-work/runs
for a in "$@"; do case "$a" in wait=*) until grep -q "^ALL_JOBS_DONE" "${a#wait=}" 2>/dev/null; do sleep 20; done ;; esac; done
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=245760 TS=0.31,0.27,0.42 B=512 DECODE_N=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
export BIGCHERRY_FEATURES=flashnext
docker stop radiance-vllm >/dev/null 2>&1
NEW=b-defon-$TAG
OFF="GGML_HIP_Q8_1_CACHE_MODE=off BIGCHERRY_RMS_Q81=0 BIGCHERRY_ACT_Q81=0 BIGCHERRY_HC_Q81=0 BIGCHERRY_SCALE_ACT_FUSE=0"
jobs=$(mktemp)
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD $NEW bigcherry:stock:linux-multi deploy-v6-plus-chunk gfx1100,gfx1201,gfx1030
VIS=0,1 SCRIPT defon-$TAG-27b tools/lab/flash-next/prod27b-ab.sh @$OLD @$NEW $R/defon-$TAG-27b
VIS=0,1 SCRIPT defon-$TAG-27b-off tools/lab/flash-next/prod27b-ab.sh @$NEW @$NEW $R/defon-$TAG-27b-off $OFF
VIS=0,1 SCRIPT defon-$TAG-27b-async tools/lab/flash-next/prod27b-ab.sh @$NEW @$NEW $R/defon-$TAG-27b-async BIGCHERRY_SCHED_ASYNC_INPUTS=1
VIS=0,1,2,3 SCRIPT defon-$TAG-flash tools/lab/flash-next/quick-ab-depth.sh 24576 @$OLD @$NEW $R/defon-$TAG-flash
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo "== 27B, no flags: A = $OLD (path off by default), B = $NEW (path on by default)"; grep -E "^d[0-9]|SERVER_FAILED|^ +[0-9]" $R/defon-$TAG-27b.log
echo "== 27B, $NEW: A = defaults (on), B = switched off explicitly"; grep -E "^d[0-9]|SERVER_FAILED|^ +[0-9]" $R/defon-$TAG-27b-off.log
echo "== 27B, $NEW: A = defaults, B = + async host inputs (1326, the one production patch still opt-in)"; grep -E "^d[0-9]|SERVER_FAILED|^ +[0-9]" $R/defon-$TAG-27b-async.log
echo "== Flash-Next 24K, flashnext profile: base = $OLD, new = $NEW"; grep -E "^base-|^new|SERVER_FAILED" $R/defon-$TAG-flash.log
md5sum $R/defon-$TAG-flash/*/*.greedy.txt 2>/dev/null | awk '{print $1}' | sort | uniq -c
echo ALL_JOBS_DONE
