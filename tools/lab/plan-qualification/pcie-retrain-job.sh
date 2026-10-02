#!/bin/bash
# Retrain degraded GPU root-port links with every GPU idle: takes the queue's host-exclusive lock (all four
# GPUs), stops radiance-vllm (R9700), runs pcie-link-check.sh --retrain (sudo password on stdin), restarts
# radiance-vllm with restart policy unless-stopped. Usage: pcie-retrain-job.sh < password
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
here=tools/lab/plan-qualification
pw=$(cat)
docker stop radiance-vllm >/dev/null 2>&1 && docker update --restart always radiance-vllm >/dev/null && echo "VLLM_STOPPED $(date -Is)"
printf '%s\n' "$pw" | BC_GPUS=0,1,2,3 bash "$here/locked-run.sh" bash "$here/pcie-link-check.sh" --retrain
rc=$?
docker start radiance-vllm >/dev/null && docker update --restart unless-stopped radiance-vllm >/dev/null && echo "VLLM_RESTARTED $(date -Is)"
echo "RETRAIN_EXIT=$rc"
exit $rc
