#!/bin/bash
# Qwen3.8-27B unsloth quant x card sweep through the queue (after the 27B patch queues).
# Needs BUILD b-3g-control (tools/lab/native-vs-patched/queue-27b-3gpu.sh). Order: dual XTX
# (-sm tensor), R9700 alone, one XTX alone. 3-arm groups use 6 pairs, 2-arm groups 4.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-27b/gguf/unsloth/Qwen3.8-27B-Q8_0.gguf
C=tools/lab/quant-sweep/configs
jobs=$(mktemp)
for dev in xtx2 r9700 xtx1; do
  case $dev in xtx2) vis=0,1 ;; r9700) vis=2 ;; xtx1) vis=0 ;; esac
  for g in 1 2 3 4; do
    arms=$(grep -c '"name"' "$C/$dev-g$g.json")
    pairs=$([ "$arms" -eq 3 ] && echo 6 || echo 4)
    echo "VIS=$vis AB qs-$dev-g$g $C/$dev-g$g.json --pairs $pairs" >> "$jobs"
  done
done
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
