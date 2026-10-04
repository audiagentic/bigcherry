#!/bin/bash
# Pin bump c061df198 -> 0504396: v5 rebuilt on the new pin (b-v5-p2b) vs the old-pin v5 binary (b-v5c, c061df198),
# ABBA x2 at ~8K and ~64K: decode + prefill t/s, greedy identity. Measures the bump itself (upstream #29825 indexer
# memory, #29824 mask construction, #29856 recurrent-state reserve) on production profile v5.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=245760 TS=0.31,0.27,0.42 B=512 DECODE_N=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
export GGML_HIP_Q8_1_CACHE_MODE=on BIGCHERRY_ROLLBACK_NO_CONT=1 BIGCHERRY_RMS_Q81=1 BIGCHERRY_ACT_Q81=1 BIGCHERRY_HC_Q81=1
export BIGCHERRY_SCALE_ACT_FUSE=1 BIGCHERRY_SCHED_ASYNC_INPUTS=1
docker stop radiance-vllm >/dev/null 2>&1
OLD=/mnt/vault/development/projects/bigcherry/workspaces/main/work/builds/c7cca11a2c58df423573227da533c2f3/e48552af8e6103ee63d041d8c943b855/bin/llama-server
jobs=$(mktemp)
R=/mnt/data/bigcherry-work/runs
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-v5-p2b bigcherry:stock:linux-multi deploy-v5 gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT bump-v5b-abba tools/lab/flash-next/abba-depths.sh $OLD @b-v5-p2b $R/flashnext-bump-v5b-abba
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
grep -E "^d[0-9]+ " $R/bump-v5b-abba.log
echo ALL_JOBS_DONE
