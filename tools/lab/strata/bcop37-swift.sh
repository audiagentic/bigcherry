#!/bin/bash
# BCOP37 unmatched lane: Strata-rocm on the model family it was written for (Swift Flash-Next GSQ-RCO IQ2_XS), to
# find out whether the garbage output on our UD-IQ4_XS file is the file or the engine. Steps, each skipped when its
# output exists: merge the shards (a layer is split across them, which the pack tool cannot read), pack, then the
# GPU smoke with prompt ids from llama-tokenize (the Strata tokenizer is a separate question).
# Usage: bcop37-swift.sh <bundle>      env: GPU (HIP index, default 2 = R9700)
set -u
B=${1:?bundle}
E=/mnt/data/bigcherry-work/external
S=/mnt/data/llm-models/strata-swift-iq2xs
M=$E/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-merged.gguf
P=$E/strata-pack-swift-iq2xs
here="$(cd "$(dirname "$0")" && pwd)"
bin=$(dirname "$(find /mnt/vault/development/projects/bigcherry/workspaces/main/work/builds -name llama-gguf-split | head -1)")
mkdir -p "$B"
if [ ! -f "$M" ]; then
  "$bin/llama-gguf-split" --merge "$S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf" "$M" > "$B/swift-merge.log" 2>&1
  echo "merge rc=$?"
fi
ls -la "$M" || exit 1
if [ ! -f "$P/index.txt" ]; then
  (cd "$E/Strata-rocm" && "$E/strata-venv/bin/python" tools/iq_pack.py --gguf "$M" --out "$P") > "$B/swift-pack.log" 2>&1
  echo "pack rc=$?"; tail -n 4 "$B/swift-pack.log" | cut -c1-300
fi
[ -f "$P/index.txt" ] || { echo "PACK_FAILED"; grep -iE "error|Traceback|unsupported" "$B/swift-pack.log" | head -8; exit 1; }
prompt='<|im_start|>user
Write a Python function that merges two sorted lists into one sorted list.<|im_end|>
<|im_start|>assistant
'
ids=$(HIP_VISIBLE_DEVICES= "$bin/llama-tokenize" -m "$M" -p "$prompt" --ids --no-bos --log-disable 2>/dev/null | tr -d '[] ' | tail -1)
echo "ids: $ids"
[ -n "$ids" ] || { echo "TOKENIZE_FAILED"; exit 1; }
IDS=$ids MAX_NEW=64 bash "$here/bcop37-smoke.sh" "$E/Strata-rocm" gfx1201 "$P" "$M" "$B/swift" --spec 2
echo SWIFT_DONE
