#!/bin/bash
# QFP15 run 3: 1316 per-node output hashes for the first 2 graph_compute calls of every context (warm-up "Hello"),
# three runs of one build: first differing node across runs = the nondeterministic op. Waits for the draft-host queue.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=245760 TS=0.31,0.27,0.42 B=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
export GGML_HIP_Q8_1_CACHE_MODE=on BIGCHERRY_ROLLBACK_NO_CONT=1 BIGCHERRY_RMS_Q81=1 BIGCHERRY_ACT_Q81=1 BIGCHERRY_HC_Q81=1
export BIGCHERRY_SCALE_ACT_FUSE=1 BIGCHERRY_DRAFT_TRACE=1 BIGCHERRY_NODE_HASH=0:2
until grep -q ALL_JOBS_DONE /mnt/data/bigcherry-work/runs/queue-draft-host.log 2>/dev/null; do sleep 30; done
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
R=/mnt/data/bigcherry-work/runs
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-det bigcherry:stock:linux-multi stock-none gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT det3-d24k tools/lab/flash-next/quick-ab-depth.sh 24576 @b-det @b-det $R/flashnext-det3-d24k
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
D=$R/flashnext-det3-d24k
grep -E "^base-|^new|SERVER_FAILED" $R/det3-d24k.log
for a in base-a new base-b; do
  grep -o "BIGCHERRY_NODE_HASH.*" $D/$a/timing.server.log | sed -E 's/ctx=0x[0-9a-f]+/ctx/' > $D/$a.nodes
  echo "$a: $(wc -l < $D/$a.nodes) node hashes"
done
for pair in "base-a new" "base-a base-b" "new base-b"; do set -- $pair
  echo "== first differing node $1 vs $2"
  paste -d'|' $D/$1.nodes $D/$2.nodes | awk -F'|' '$1 != $2 {print NR": "$1; print NR": "$2; exit}'
done
echo ALL_JOBS_DONE
