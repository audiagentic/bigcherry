#!/bin/bash
# RR01: make an 8-bit Qwen3.8-27B container radiance serves from the bf16 GGUF on Brutus. radiance has no plugin for
# a GGUF Q8_0 27B ("qwen35 is served at: (unquantised), fp8_e4m3"), so the Q8_0 file cannot be imported as it is.
#   fp8  block fp8 (E4M3, one bf16 scale a 128x128 block), the form of Qwen's own FP8 release, with radiance's recipe
#        of record (data/recipes/q38-27b-fp8block-df2.recipe)
#   i8   int8 codes with one bf16 scale a row a 128 columns (recipes/q38-27b-i8-df2.recipe here): the form whose inner
#        product is the iu8 WMMA, which gfx11 has
# First with the DFlash2 drafter merged; if the drafter checkpoint is not accepted, again without it.
# Start it directly (it queues its own jobs on the gfx1201 card):
#   bash tools/lab/radiance/convert-27b-8bit.sh <fp8|i8>
# env: R (runs root), OUT_DIR, BF16 (the bf16 checkpoint), DRAFT_DIR, BUILD (b-main2)
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=${R:-/mnt/data/bigcherry-work/runs}
src=/mnt/data/bigcherry-work/engines/radiance
form=${1:?fp8 or i8}
case $form in
    fp8) recipe=$src/data/recipes/q38-27b-fp8block-df2.recipe ;;
    i8) recipe=$PWD/tools/lab/radiance/recipes/q38-27b-i8-df2.recipe ;;
    *) echo "unknown form $form"; exit 2 ;;
esac
out_dir=${OUT_DIR:-/mnt/data/llm-models/radiance/bigcherry/qwen3.8-27b-$form}
bf16=${BF16:-/mnt/data/llm-models/qwen3.8-27b/gguf/unsloth/BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf}
draft=${DRAFT_DIR:-/mnt/data/llm-models/qwen3.8-27b/Qwen3.8-27B-DFlash2-FP8}
export INPUT=$bf16 RECIPE=$recipe
export SETS="quantization_config.quant_method=fp8;quantization_config.fmt=e4m3;quantization_config.weight_block_size=128 128;quantization_config.activation_scheme=dynamic"
jobs=$(mktemp)
attempt() {  # <run name> <out file> [DRAFT]
    rm -f "$R/$1.log" "$2"
    echo "VIS=2 SCRIPT $1 tools/lab/radiance/rad-convert.sh @${BUILD:-b-main2} $2" > "$jobs"
    DRAFT=${3:-} bash tools/lab/plan-qualification/queue.sh "$jobs" > /dev/null 2>&1
    grep -vE "PCIe" "$R/$1.log" | cut -c1-230 | head -24
    grep -q "^SCRIPT_EXIT=0" "$R/$1.log"
}
if attempt "cv-$form-df2" "$out_dir/qwen3.8-27b-$form-df2.rad" "$draft"; then
    echo "CONVERTED with the drafter"
elif attempt "cv-$form" "$out_dir/qwen3.8-27b-$form.rad"; then
    echo "CONVERTED without the drafter"
else
    echo "CONVERSION FAILED"
fi
rm -f "$jobs"
echo CONVERT_27B_DONE
