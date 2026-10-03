#!/bin/bash
# 1292 parity at 80K, now measurable with 1294: base = 1291+1294, new = 1291+1292+1294 (b-flash-topk-1),
# no MTP, each binary twice at ~10K and ~80K (kpool-parity.sh).
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
new=/mnt/vault/development/projects/bigcherry/workspaces/main/work/builds/090aa314b2d5e250727d572d7d80441e/841024fa50a66cd35746f80486cb6c43/bin/llama-server
docker stop radiance-vllm >/dev/null 2>&1 && docker update --restart always radiance-vllm >/dev/null
jobs=$(mktemp)
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-flash-topkbase-1 bigcherry:stock:linux-multi ar-cpu-root-topk gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT flashnext-kpool-parity-2 tools/lab/flash-next/kpool-parity.sh @b-flash-topkbase-1 $new /mnt/data/bigcherry-work/runs/flashnext-kpool-parity-2
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
