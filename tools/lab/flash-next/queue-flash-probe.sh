#!/bin/bash
# QFN01 Flash-Next layout probe on all four GPUs. Stops radiance-vllm (R9700) for the run and restarts it
# (restart policy back to unless-stopped). Launch behind a waiter so vLLM is only down while this runs.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
# Contract sessions append evidence to patches/*/evidence/validation.json in this checkout, which makes
# BUILD rows refuse a dirty tree. Save those changes (append-only evidence, copied back to the repo
# by hand) and restore the committed files before building.
bk=/mnt/data/bigcherry-work/evidence-backup/$(date +%Y%m%dT%H%M%S)
for f in $(git status --porcelain -- 'patches/*/evidence/validation.json' | awk '{print $2}'); do
  mkdir -p "$bk/$(dirname "$f")" && cp "$f" "$bk/$f" && git checkout -- "$f" && echo "saved evidence $f -> $bk/$f"
done
docker stop radiance-vllm >/dev/null 2>&1 && docker update --restart always radiance-vllm >/dev/null
jobs=$(mktemp)
cat > "$jobs" <<JOBS
VIS=0,1,2,3 SCRIPT flashnext-layer-c061 tools/lab/flash-next/layout-probe.sh @b-flash-c061 /mnt/data/bigcherry-work/runs/flashnext-layer-c061/out cpu3-layer
VIS=0,1,2,3 SCRIPT flashnext-layer-1280 tools/lab/flash-next/layout-probe.sh @b-flash-mtp2 /mnt/data/bigcherry-work/runs/flashnext-layer-1280/out cpu3-layer
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
docker start radiance-vllm >/dev/null && docker update --restart unless-stopped radiance-vllm >/dev/null && echo "VLLM_RESTARTED $(date -Is)"
echo ALL_JOBS_DONE
