#!/bin/bash
# QFP17 / 1335 (upstream #29901 tiled lightning indexer): build the production set + 1335, then
#   1. backend correctness vs CPU, flag off and on, with the activation marker (indexer-backend-test.sh);
#   2. per depth a Flash-Next prefill ABBA on the one binary, A = default (tile on), B = BIGCHERRY_INDEXER_TILE=0;
#   3. fidelity at the first depth: D = tile on, S = tile off (flash-fidelity.sh).
# Usage: queue-indexer-tile.sh <tag> <depth>... [wait=<log with ALL_JOBS_DONE>]
set -u
TAG=${1:?tag}; shift
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=/mnt/data/bigcherry-work/runs
depths=()
for a in "$@"; do case "$a" in wait=*) until grep -q "^ALL_JOBS_DONE" "${a#wait=}" 2>/dev/null; do sleep 20; done ;; *) depths+=("$a") ;; esac; done
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=245760 TS=0.31,0.27,0.42 UB=512 B=512 DECODE_N=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
export BIGCHERRY_FEATURES=flashnext
docker stop radiance-vllm >/dev/null 2>&1
RUN=b-ixtile-$TAG
jobs=$(mktemp)
echo "VIS=0,1,2,3 BUILD $RUN bigcherry:stock:linux-multi stock-none gfx1100,gfx1201,gfx1030" > "$jobs"
echo "VIS=0,1,2,3 SCRIPT ixtile-$TAG-backend tools/lab/flash-next/indexer-backend-test.sh @$RUN $R/ixtile-$TAG-backend" >> "$jobs"
for d in "${depths[@]}"; do
  echo "VIS=0,1,2,3 SCRIPT ixtile-$TAG-d$d tools/lab/flash-next/flash-prefill-env-ab.sh $d @$RUN $R/ixtile-$TAG-d$d BIGCHERRY_INDEXER_TILE=0" >> "$jobs"
done
echo "VIS=0,1,2,3 SCRIPT ixtile-$TAG-fid tools/lab/flash-next/flash-fidelity.sh ${depths[0]} @$RUN $R/ixtile-$TAG-fid noref BIGCHERRY_INDEXER_TILE=0" >> "$jobs"
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo "== backend test"; grep -E "^tile=|^perf|LIGHTNING_INDEXER|tests passed|FAIL|BUILD_FAILED" $R/ixtile-$TAG-backend.log | cut -c1-220
echo "== prefill: A = tile on (default), B = BIGCHERRY_INDEXER_TILE=0"
for d in "${depths[@]}"; do grep -E "^d[0-9]|^md5|SERVER_FAILED" $R/ixtile-$TAG-d$d.log; done
echo "== fidelity: D = tile on, S = tile off"; grep -E "^D:|^S:|^D2:| vs |SERVER_FAILED" $R/ixtile-$TAG-fid.log
echo ALL_JOBS_DONE
