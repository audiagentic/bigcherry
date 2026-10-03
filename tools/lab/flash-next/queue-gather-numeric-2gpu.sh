#!/bin/bash
# 1295 numeric check on 2 XTX (-ts 1,1: one KV head per rank), experts of layers 30-47 on CPU, f16 KV.
# kernel quantizes Q to q8_1), so remaining probability differences isolate the gather itself.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export CTK=f16 CTV=f16 DEPTHS=8192 CTX=16384 DEVS=ROCm0,ROCm1 TS=1,1 EXTRA_OT='blk.(3[0-9]|4[0-7]).ffn_(up|gate|down)_exps.weight=CPU'
docker stop radiance-vllm >/dev/null 2>&1 && docker update --restart always radiance-vllm >/dev/null
jobs=$(mktemp)
echo "VIS=0,1,2,3 SCRIPT flashnext-gather-numeric-2gpu tools/lab/flash-next/gather-numeric.sh /mnt/vault/development/projects/bigcherry/workspaces/main/work/builds/cb765fc5ba3f516f1cfd5d170b18642a/3a9a890b50c7321e7d114e03fc8b2aaa/bin/llama-server /mnt/data/bigcherry-work/runs/flashnext-gather-numeric-2gpu" > "$jobs"
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
