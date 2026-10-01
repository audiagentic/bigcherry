#!/bin/bash
# Contract-campaign batch for the remaining 27B-track patches on Brutus gfx1100 (device 0), via the
# cached/content-addressed plan-qualification queue: 1254 (nro05), 1206 (rd13), 1263 (prbe41), 4 sessions each,
# patches interleaved per session so consecutive sessions of one patch are separated by other work (PA35 cooldown).
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
M=/mnt/data/llm-models
Q4=$M/qwen3.5-4B/gguf/mtp/Qwen3.5-4B-UD-Q6_K_XL.gguf
Q35=$M/qwen3.6-35B-A3B/gguf/mtp/Qwen3.6-35B-A3B-APEX-MTP-I-Compact.gguf
GO=$M/gpt-oss-20B/gguf/gpt-oss-20b-UD-Q6_K_XL.gguf
CORPUS=tools/bigcherry/bench/corpora/mtp-27b-v1.jsonl
export BC_MODEL=$Q4
jobs=$(mktemp)
for s in 1 2 3 4; do
  echo "MODEL=$Q35 1254_nro05_gdn_mtp_prefix_tail 1254_nro05_gdn_mtp_prefix_tail/nro05 gfx1100 0 t-1254-gfx1100-s$s --common-patches 1253_nro04_gfx1100_bf16_chunked_gdn --producer-corpus $CORPUS" >> "$jobs"
  echo "MODEL=$Q4 1206_rd13_mul_mat_add_view_fusion 1206_rd13_mul_mat_add_view_fusion/rd13 gfx1100 0 t-1206-gfx1100-s$s --producer-input control_model=$GO" >> "$jobs"
  echo "MODEL=$Q4 1263_prbe41_ssm_conv_channels_major 1263_prbe41_ssm_conv_channels_major/prbe41 gfx1100 0 t-1263-gfx1100-s$s --producer-input control_model=$GO" >> "$jobs"
done
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
