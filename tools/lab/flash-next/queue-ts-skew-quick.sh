#!/bin/bash
# RV4214 step 0: zero-code tensor-split rebalance screens on the deployment candidate (production -ts 2,2,3 vs
# nudges toward/away from the R9700) -- tests whether the ~121 us cpu-root arrival skew is a decode lever.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 TS=2,2,3 CTX=${CTX:-131072}  # both arms at reduced context so heavier splits fit
base=/mnt/vault/development/projects/bigcherry/workspaces/main/work/builds/1616483f7592d564a3882e8f17330996/f08f543aee27491e58d23986fbfb08bb/bin/llama-server
docker stop radiance-vllm >/dev/null 2>&1
for ts in ${TS_LIST:-2,2,3.4 2.1,2.1,2.8 1.9,2.1,3 2,2,3.8 2.2,2.2,2.6}; do
  echo "== ts $ts"
  bash tools/lab/flash-next/quick-ab.sh $base $base /mnt/data/bigcherry-work/runs/flashnext-ts-${ts//,/_} TS=$ts
done
echo ALL_JOBS_DONE
