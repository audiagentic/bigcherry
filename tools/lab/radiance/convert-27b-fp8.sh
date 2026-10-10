#!/bin/bash
# RR01: make the 8-bit Qwen3.8-27B container radiance serves - block fp8 (E4M3, one bf16 scale a 128x128 block), the
# form of Qwen's own FP8 release - from the bf16 GGUF on Brutus with radiance's recipe of record
# (data/recipes/q38-27b-fp8block-df2.recipe). radiance has no plugin for a GGUF Q8_0 27B ("qwen35 is served at:
# (unquantised), fp8_e4m3"), so the Q8_0 file cannot be imported as it is.
# First with the DFlash2 drafter merged; if the drafter checkpoint is not accepted, again without it.
# Start it directly (it queues its own jobs on the gfx1201 card):
#   bash tools/lab/radiance/convert-27b-fp8.sh
# env: R (runs root), OUT_DIR, BF16 (the bf16 checkpoint), DRAFT_DIR, BUILD (b-main2)
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=${R:-/mnt/data/bigcherry-work/runs}
src=/mnt/data/bigcherry-work/engines/radiance
out_dir=${OUT_DIR:-/mnt/data/llm-models/radiance/bigcherry/qwen3.8-27b-fp8}
bf16=${BF16:-/mnt/data/llm-models/qwen3.8-27b/gguf/unsloth/BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf}
draft=${DRAFT_DIR:-/mnt/data/llm-models/qwen3.8-27b/Qwen3.8-27B-DFlash2-FP8}
export INPUT=$bf16 RECIPE=$src/data/recipes/q38-27b-fp8block-df2.recipe
export SETS="quantization_config.quant_method=fp8;quantization_config.fmt=e4m3;quantization_config.weight_block_size=128 128;quantization_config.activation_scheme=dynamic"
jobs=$(mktemp)
attempt() {  # <run name> <out file> [DRAFT]
    rm -f "$R/$1.log" "$2"
    echo "VIS=2 SCRIPT $1 tools/lab/radiance/rad-convert.sh @${BUILD:-b-main2} $2" > "$jobs"
    DRAFT=${3:-} bash tools/lab/plan-qualification/queue.sh "$jobs" > /dev/null 2>&1
    grep -vE "PCIe" "$R/$1.log" | cut -c1-230 | head -24
    grep -q "^SCRIPT_EXIT=0" "$R/$1.log"
}
if attempt cv-fp8-df2 "$out_dir/qwen3.8-27b-fp8-df2.rad" "$draft"; then
    echo "CONVERTED with the drafter"
elif attempt cv-fp8 "$out_dir/qwen3.8-27b-fp8.rad"; then
    echo "CONVERTED without the drafter"
else
    echo "CONVERSION FAILED"
fi
rm -f "$jobs"
echo CONVERT_27B_FP8_DONE
