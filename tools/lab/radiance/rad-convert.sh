#!/bin/bash
# RR01: convert a checkpoint (GGUF, safetensors or a checkpoint directory) to a radiance container with rad-convert,
# keeping the checkpoint's own weight encoding unless QUANT rules are given, and report whether every op resolved to a
# kernel. Declare is run against radiance's own library on the gfx12 card, so the container is the one radiance
# itself would produce; serving it through libr3 on gfx11 is a separate step.
# Run as a queue SCRIPT job on the gfx1201 card:
#   VIS=2 SCRIPT cv-q8 tools/lab/radiance/rad-convert.sh @<any build run> <out.rad>
# The first argument (a llama-server path from the queue) is ignored.
# Usage: rad-convert.sh <ignored> <out.rad>
# env: INPUT (checkpoint; default the Qwen3.8-27B Q8_0 GGUF), DRAFT (a drafter checkpoint directory to merge),
#      RECIPE (a recipe file), SETS (metadata keys K=V separated by ';'), QUANT (--quant rules, separated by ';'), GPU (HIP index of the gfx12 card, 2), KERNELS
#      (libr4d:libref), RADIANCE_SRC, CONVERT_TIMEOUT (7200 s)
set -u
out=$2
mkdir -p "$(dirname "$out")"
src=${RADIANCE_SRC:-/mnt/data/bigcherry-work/engines/radiance}
input=${INPUT:-/mnt/data/llm-models/qwen3.8-27b/gguf/unsloth/Qwen3.8-27B-Q8_0.gguf}
export ROCM_PATH=${ROCM_PATH:-/opt/rocm-7.2.4}
export PATH="$ROCM_PATH/bin:$PATH"
for v in $(env | grep -oE "^(BIGCHERRY_[A-Z0-9_]+|GGML_HIP_[A-Z0-9_]+)"); do unset "$v"; done
opts=(--home "$src/build/radiance_home" --kernels "${KERNELS:-libr4d:libref}")
[ -n "${DRAFT:-}" ] && opts+=(--draft-model "$DRAFT")
[ -n "${RECIPE:-}" ] && opts+=(--recipe "$RECIPE")
if [ -n "${SETS:-}" ]; then  # metadata keys, K=V separated by ';' (a value may hold spaces)
    IFS=';' read -ra sets <<< "$SETS"
    for kv in "${sets[@]}"; do opts+=(--set "$kv"); done
fi
if [ -n "${QUANT:-}" ]; then
    IFS=';' read -ra rules <<< "$QUANT"
    for rule in "${rules[@]}"; do opts+=(--quant "$rule"); done
fi
echo "radiance $(git -C "$src" rev-parse --short HEAD); input $(basename "$input") ($(du -sh "$input" | cut -f1)); out $out"
echo "free on $(dirname "$out"): $(df -h "$(dirname "$out")" | tail -1 | awk '{print $4}')"
t0=$(date +%s)
HIP_VISIBLE_DEVICES=${GPU:-2} timeout "${CONVERT_TIMEOUT:-7200}" "$src/build/bin/rad-convert" "${opts[@]}" "$input" -o "$out" > "$out.convert.log" 2>&1
rc=$?
echo "rad-convert: exit $rc in $(( $(date +%s) - t0 )) s"
grep -E "^(E|W) |error|refus|no kernel|unresolved|not resolve|encoding|quantis" "$out.convert.log" | sort | uniq -c | sort -rn | head -24 | cut -c1-230
tail -6 "$out.convert.log" | cut -c1-230
[ -s "$out" ] && ls -la --block-size=G "$out" | awk '{print "container", $5, $9}'
exit $rc
