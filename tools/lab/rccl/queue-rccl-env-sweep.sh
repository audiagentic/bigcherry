#!/bin/bash
# Queue the dual-XTX RCCL env sweep (R9700/vLLM untouched) on the active pin's production composition
# (BUILD rows build the bigcherry source: serving-core + upstream-fixes + validated-enhancements, which
# includes the promoted adaptive AllReduce closure), with llama-bench.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
jobs=$(mktemp)
cat > "$jobs" <<JOBS
VIS=0,1 BUILD b-stock-c061-bench bigcherry:stock:linux-multi - gfx1100 bin/llama-bench
VIS=0,1 SCRIPT rccl-env-sweep-stock tools/lab/rccl/rccl-env-sweep.sh @b-stock-c061-bench /mnt/data/bigcherry-work/runs/rccl-env-sweep-stock/out
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE

