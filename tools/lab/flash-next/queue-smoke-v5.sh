#!/bin/bash
# v5 on the new pin (0504396) on other models: 27B, Gemma-4, gpt-oss, flags off vs on (incl. 1312/1313/1326), greedy
# identity per model. Waits for the 1327 screen.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
# queue.sh requires BC_MODEL; smoke-models.sh picks its own models
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
R=/mnt/data/bigcherry-work/runs
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-v5-p2c bigcherry:stock:linux-multi deploy-v5 gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT smoke-v5-p2b tools/lab/flash-next/smoke-models.sh @b-v5-p2c $R/smoke-v5-p2b
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
tail -20 $R/smoke-v5-p2b.log
echo ALL_JOBS_DONE
