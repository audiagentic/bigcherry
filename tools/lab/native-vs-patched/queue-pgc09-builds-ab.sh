#!/bin/bash
# PGC09: production-flag MTP A/B on the exact t-0840d contract builds (validated BC + 0860 + 1225
# with RCCL, vs + 0840 adaptive with exact-f32 host path), provider auto on both, depth 5, ubatch 2048,
# 1-token prompts with long generation. Separates adaptive's decode effect from the contract lane's
# prompt-sensitive MTP acceptance (t-0840d-s1: -6% wall, acceptance 50.6 vs 54.0%).
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm BC_MODEL=/mnt/vault/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
L=tools/lab/native-vs-patched
jobs=$(mktemp)
cat > "$jobs" <<JOBS
AB ab-27b-pgc09-builds $L/server-ab-pgc09-builds.json --pairs 6
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
python3 tools/lab/ar-accuracy/gates.py acceptance /mnt/data/bigcherry-work/runs/ab-27b-pgc09-builds/result
rm -f "$jobs"
echo ALL_JOBS_DONE
