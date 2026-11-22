#!/usr/bin/env bash
set -euo pipefail
# Required environment: BC_MODEL_ROOT, ROCM_PATH, FFMPEG_LIB (host paths are not recorded here).
# Usage: record.sh RUN_DIR RADIANCE_RELEASE_DIR CORPUS_FILE (RUN_DIR must exist; the reference is written to RUN_DIR/reference).
: "${BC_MODEL_ROOT:?}" "${ROCM_PATH:?}" "${FFMPEG_LIB:?}"
r=$1
old=$2
corpus=$3
model=${BC_MODEL_ROOT}/radiance/StillDeadcode/qwen3.8-next-flash-fp8-iq4r-moe/qwen3.8-next-flash-fp8-iq4r-moe.rad
trap 'code=$?; echo "$code" > "$r/exit-code"' EXIT
sha256sum "$old/bin/radiance" "$corpus" > "$r/inputs.sha256"
rocm-smi --showbus --showproductname > "$r/gpu-inventory.log"
grep -q 'GPU\[0\].*gfx1100' "$r/gpu-inventory.log"
grep -q 'GPU\[1\].*gfx1100' "$r/gpu-inventory.log"
env -i LD_LIBRARY_PATH=${FFMPEG_LIB}:${ROCM_PATH}/lib PATH=/usr/bin:/bin:${ROCM_PATH}/bin ROCM_PATH=${ROCM_PATH} HIP_PATH=${ROCM_PATH} \
  ROCR_VISIBLE_DEVICES=0,1 HIP_VISIBLE_DEVICES=0,1 "$old/bin/radiance" --model "$model" \
  --radiance-home "$old/radiance_home" --kernels libr3,libr3moe,libref --placement expert_tiered \
  --tp 2 --tp-wire exact --host-pool-mib 14336 --gpu-headroom-mib 320 --kv-cache-dtype fp8 \
  --max-model-len 8192 --max-num-seqs 1 --max-num-batched-tokens 64 \
  --kld-record "$r/reference" --kld-corpus "$corpus" > "$r/record.log" 2>&1
