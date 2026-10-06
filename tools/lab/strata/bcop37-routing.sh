#!/bin/bash
# MET01 / BCOP37: routing traces of the Swift IQ2_XS file for three request types (prose = repository docs, code =
# repository Python, json = planning/ledger JSON), written by Strata's --dump-routing, then scored by
# routing-skew.py (frequency placement against whole layers, transfer between request types, prefill upload set).
# Each prompt is <depth> ids of the material wrapped as one user turn, plus MAX_NEW generated tokens.
# Writes <bundle>/routing-<type>.{ids,bin,log} and <bundle>/routing-skew.txt. Needs the pack made by bcop37-swift.sh.
# Usage: bcop37-routing.sh <bundle> [depth=4096]      env: GPU (HIP index, default 2), MAX_NEW (256)
set -u
B=${1:?bundle}; d=${2:-4096}
E=/mnt/data/bigcherry-work/external
M=$E/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-merged.gguf
P=$E/strata-pack-swift-iq2xs
T=$E/Strata-rocm
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
bin=$(dirname "$(find "$repo/work/builds" -name llama-tokenize | head -1)")
export LD_LIBRARY_PATH=$E/strata-rocm-shim/lib PATH=$E/strata-rocm-shim/bin:$PATH
tok() { HIP_VISIBLE_DEVICES= "$bin/llama-tokenize" -m "$M" "$@" --ids --no-bos --log-disable 2>/dev/null | tr -d '[] ' | tail -1; }
cat "$repo"/docs/reference/*/*.md > "$B/routing-prose.text"
cat "$repo"/tools/bigcherry/patch/*.py > "$B/routing-code.text"
cat "$repo"/docs/releases/CURRENT_RELEASE_LEDGER.ndjson "$repo"/.audiagentic/runtime/ledger/fragments/*.json 2>/dev/null > "$B/routing-json.text"
head=$(tok -p '<|im_start|>user
Review the following material and report what matters.

')
tail=$(tok -p '<|im_end|>
<|im_start|>assistant
')
args=()
for t in prose code json; do
  body=$(tok -f "$B/routing-$t.text")
  echo "$head,$(echo "$body" | cut -d, -f1-"$d"),$tail" > "$B/routing-$t.ids"
  rm -f "$B/routing-$t.bin"
  HIP_VISIBLE_DEVICES=${GPU:-2} "$T/build_gfx1201/strata" --pack "$P" --native "$M" --ple-gguf "$M" --tokens-file "$B/routing-$t.ids" \
    --max-new ${MAX_NEW:-256} --max-context $((d + 2048)) --stats --spec 2 --prefill auto --expert-cache auto \
    --expert-profile "$T/data/expert-profile.bin" --dump-routing "$B/routing-$t.bin" > "$B/routing-$t.log" 2>&1
  echo "$t rc=$? ids $(tr ',' '\n' < "$B/routing-$t.ids" | wc -l) trace $(stat -c %s "$B/routing-$t.bin" 2>/dev/null) bytes; $(grep -aE '^(decode|prefill) ' "$B/routing-$t.log" | tr -s ' ' | tr '\n' ';')"
  args+=("$t=$B/routing-$t.bin")
done
python3 "$here/routing-skew.py" --profile "$T/data/expert-profile.bin" "${args[@]}" | tee "$B/routing-skew.txt"
echo ROUTING_DONE
