#!/bin/bash
# RR01/RR04: record a rad-kbench fixture from a real container, so the kernels that container actually uses are
# checked and timed (the shipped kernels.rkb comes from an fp8 model and never reaches the int8, 4-bit or MoE rows).
# Recorded with radiance's own libr4d on the gfx12 card: the graph and shapes are the container's, the weights are
# drawn, and the answers come from the reference library, so the fixture replays on any card.
# Run as a queue SCRIPT job on the gfx1201 card:
#   VIS=2 SCRIPT fx-mxfp4 tools/lab/radiance/record-fixture.sh @<any build run> <out.rkb>
# The first argument (a llama-server path from the queue) is ignored.
# Usage: record-fixture.sh <ignored> <out.rkb>
# env: MODEL (.rad container), GPU (HIP index of the gfx12 card, 2 on Brutus), MAX_CASES (1), SPEC (container's own),
#      KV (bf16), RADIANCE_SRC
set -u
out=$2
mkdir -p "$(dirname "$out")"
src=${RADIANCE_SRC:-/mnt/data/bigcherry-work/engines/radiance}
model=${MODEL:-/mnt/data/llm-models/radiance/StillDeadcode/qwen3.8-27b-mxfp4/qwen3.8-27b-mxfp4.rad}
export ROCM_PATH=${ROCM_PATH:-/opt/rocm-7.2.4}
export PATH="$ROCM_PATH/bin:$PATH"
for v in $(env | grep -oE "^(BIGCHERRY_[A-Z0-9_]+|GGML_HIP_[A-Z0-9_]+)"); do unset "$v"; done
opts=(--max-cases "${MAX_CASES:-1}" --kv-cache-dtype "${KV:-bf16}")
[ -n "${SPEC:-}" ] && opts+=(--num-speculative-tokens "$SPEC")
echo "radiance $(git -C "$src" rev-parse --short HEAD); recording from $(basename "$model") on HIP device ${GPU:-2}"
HIP_VISIBLE_DEVICES=${GPU:-2} RADIANCE_HOME="$src/build/radiance_home" timeout "${RECORD_TIMEOUT:-2400}" \
    "$src/build/bin/rad-kbench" -m "$model" --kernels libr4d,libref "${opts[@]}" --record "$out" > "$out.log" 2>&1
rc=$?
echo "rad-kbench --record: exit $rc"
grep -E "device:|case\(s\)|checked,|recorded|skipped|refused" "$out.log" | sort | uniq -c | sort -rn | head -20 | cut -c1-220
[ -s "$out" ] || { echo "NO_FIXTURE"; tail -12 "$out.log" | cut -c1-220; exit 1; }
ls -la --block-size=M "$out" | awk '{print "fixture", $5, $9}'
exit $rc
