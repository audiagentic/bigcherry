#!/bin/bash
# BCOP37 step 4: prove the Strata-rocm build executes on the intended GPU before any benchmark. One greedy
# generation on a fixed prompt, with VRAM sampled while it runs, the output decoded, and the raw log kept.
# Writes <bundle>/gpu-smoke.txt and gpu-smoke.raw.log. Changes nothing in the Strata tree.
# Usage: bcop37-smoke.sh <strata tree> <gfx target> <pack dir> <model gguf> <bundle> [extra strata args...]
#        env: GPU (HIP index, default 2 = R9700), MAX_NEW (48), MAX_CONTEXT (4096), ROCM (shim root), PY (venv python)
set -u
T=${1:?strata tree}; ARCH=${2:?gfx target}; P=${3:?pack dir}; M=${4:?model gguf}; B=${5:?bundle}; shift 5
ROCM=${ROCM:-/mnt/data/bigcherry-work/external/strata-rocm-shim}
PY=${PY:-/mnt/data/bigcherry-work/external/strata-venv/bin/python}
gpu=${GPU:-2}
export LD_LIBRARY_PATH=$ROCM/lib PATH=$ROCM/bin:$PATH
mkdir -p "$B"
prompt='<|im_start|>user
Write a Python function that merges two sorted lists into one sorted list.<|im_end|>
<|im_start|>assistant
'
ids=$(cd "$T/tools" && PROMPT="$prompt" "$PY" -c '
import os, pathlib, sys
sys.path.insert(0, ".")
from strata_tokenizer import Tokenizer
tk = Tokenizer.from_pack(pathlib.Path(sys.argv[1]) / "tokenizer")
print(",".join(str(i) for i in tk.encode(os.environ["PROMPT"], parse_special=True)))' "$P")
[ -n "$ids" ] || { echo "SMOKE_FAILED: tokenizer produced no ids"; exit 1; }
vram() { rocm-smi --showmeminfo vram 2>/dev/null | grep "GPU\[$gpu\].*Total Used" | awk '{print int($NF/1048576)}'; }
before=$(vram)
cmd=("$T/build_$ARCH/strata" --pack "$P" --native "$M" --ple-gguf "$M" --tokens "$ids" --max-new ${MAX_NEW:-48}
     --max-context ${MAX_CONTEXT:-4096} --stats "$@")
echo "HIP_VISIBLE_DEVICES=$gpu ${cmd[*]}" >> "$B/commands.txt"
( peak=0; while sleep 1; do v=$(vram); [ "${v:-0}" -gt "$peak" ] && { peak=$v; echo "$peak" > "$B/gpu-smoke.vram-peak"; }; done ) &
sampler=$!
start=$(date +%s.%N)
env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=$gpu timeout ${TIMEOUT:-1800} "${cmd[@]}" > "$B/gpu-smoke.raw.log" 2>&1
rc=$?
end=$(date +%s.%N)
kill $sampler 2>/dev/null
out_ids=$(grep -m1 "^output" "$B/gpu-smoke.raw.log" | sed 's/^output *: *//')
{
  echo "date: $(date -Is)"; echo "gpu: HIP index $gpu ($(rocm-smi --showproductname 2>/dev/null | grep "GPU\[$gpu\].*GFX Version" | awk '{print $NF}'))"
  echo "rc: $rc   wall: $(echo "$end - $start" | bc) s"
  echo "prompt tokens: $(echo "$ids" | tr ',' '\n' | wc -l)"
  echo "vram before: ${before} MiB   peak while running: $(cat "$B/gpu-smoke.vram-peak" 2>/dev/null) MiB"
  echo "output ids: $out_ids"
  echo "output text:"; [ -n "$out_ids" ] && (cd "$T/tools" && "$PY" strata_tokenizer.py --pack "$P" --ids "$out_ids")
  echo; echo "engine lines:"; grep -iE "device|gfx|hip|expert|cache|tok/s|ms/token|prefill|error|fail|warn|pool|vram|arena" "$B/gpu-smoke.raw.log" | head -60 | cut -c1-220
} > "$B/gpu-smoke.txt" 2>&1
cat "$B/gpu-smoke.txt"
echo "--- last lines of the raw log"; tail -15 "$B/gpu-smoke.raw.log" | cut -c1-220
[ $rc -eq 0 ] && echo SMOKE_DONE || echo "SMOKE_FAILED rc=$rc"
