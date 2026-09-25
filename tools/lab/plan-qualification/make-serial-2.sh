#!/bin/bash
# Write the second plan-qualification batch (host paths from the environment).
# Usage: make-serial-2.sh <out-jobs-file>
# Needs: BC_MODEL_ROOT (model store), BC_HIP_PATH (default ROCm campaign
# prefix), BC_HIP_10 (ROCm 10.x campaign prefix, for HIP >= 7.15 patches),
# BC_CORPUS (MTP prompt corpus jsonl).
set -eu
out=$1
: "${BC_MODEL_ROOT:?}" "${BC_HIP_10:?}" "${BC_CORPUS:?}"
Q4=$BC_MODEL_ROOT/qwen3.5-4B/gguf/mtp/Qwen3.5-4B-UD-Q6_K_XL.gguf
Q35=$BC_MODEL_ROOT/qwen3.6-35B-A3B/gguf/mtp/Qwen3.6-35B-A3B-APEX-MTP-I-Compact.gguf
GO=$BC_MODEL_ROOT/gpt-oss-20B/gguf/gpt-oss-20b-UD-Q6_K_XL.gguf
Q27=$BC_MODEL_ROOT/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
: > "$out"
for s in 1 2 3 4; do
  for a in "gfx1100 0" "gfx1201 2"; do
    set -- $a
    echo "MODEL=$Q4 1262_nro15_mmvdq 1262_nro15_mmvdq/nro15 $1 $2 t-1262-$1-s$s" >> "$out"
    echo "MODEL=$Q4 1253_nro04_gfx1100_bf16_chunked_gdn 1253_nro04_gfx1100_bf16_chunked_gdn/nro04 $1 $2 t-1253-$1-s$s" >> "$out"
    echo "MODEL=$Q35 1254_nro05_gdn_mtp_prefix_tail 1254_nro05_gdn_mtp_prefix_tail/nro05 $1 $2 t-1254-$1-s$s --common-patches 1253_nro04_gfx1100_bf16_chunked_gdn --producer-corpus $BC_CORPUS" >> "$out"
    echo "MODEL=$Q4 1263_prbe41_ssm_conv_channels_major 1263_prbe41_ssm_conv_channels_major/prbe41 $1 $2 t-1263-$1-s$s --producer-input control_model=$GO" >> "$out"
    echo "HIP=$BC_HIP_10 MODEL=$Q35 1256_nro07_topk_hybrid 1256_nro07_topk_hybrid/nro07 $1 $2 t-1256-$1-s$s --producer-input control_model=$Q4" >> "$out"
    echo "HIP=$BC_HIP_10 MODEL=$Q35 1257_nro08_topk_wave32 1257_nro08_topk_wave32/nro08 $1 $2 t-1257-$1-s$s --common-patches 1256_nro07_topk_hybrid --producer-input control_model=$Q4" >> "$out"
  done
  # 1241 needs both gfx1100 cards (-sm tensor MTP lane on the 27B Q8_0 model).
  echo "VIS=0,1 MODEL=$Q27 1241_rd33_mmvq_q8_0_f32_decode 1241_rd33_mmvq_q8_0_f32_decode/rd33 gfx1100 0,1 t-1241-gfx1100-s$s --producer-input control_model=$Q4 --producer-corpus $BC_CORPUS" >> "$out"
done
# 1205 gfx1201 lost its 4th session to the full disk.
echo "MODEL=$Q4 1205_rd12_paired_mmvq_dual_output 1205_rd12_paired_mmvq_dual_output/rd12 gfx1201 2 t-1205-gfx1201-s4" >> "$out"
wc -l "$out"
