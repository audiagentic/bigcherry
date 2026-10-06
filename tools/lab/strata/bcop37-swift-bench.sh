#!/bin/bash
# BCOP37 unmatched lane, Strata side: long-prompt prefill and decode speed of Strata-rocm on the Swift IQ2_XS file,
# one GPU (R9700). The prompt is repository documentation tokenized by llama-tokenize and cut to <depth> ids, wrapped
# as one user turn asking for a summary, so the BigCherry control can be given the same ids.
# Writes <bundle>/swift-bench-d<depth>.{ids,raw.log}. Needs the pack made by bcop37-swift.sh.
# Usage: bcop37-swift-bench.sh <bundle> <depth>...     env: GPU (HIP index, default 2), MAX_NEW (256), REPS (2)
set -u
B=${1:?bundle}; shift
E=/mnt/data/bigcherry-work/external
M=$E/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-merged.gguf
P=$E/strata-pack-swift-iq2xs
T=$E/Strata-rocm
repo="$(cd "$(dirname "$0")/../../.." && pwd)"
bin=$(dirname "$(find "$repo/work/builds" -name llama-tokenize | head -1)")
export LD_LIBRARY_PATH=$E/strata-rocm-shim/lib PATH=$E/strata-rocm-shim/bin:$PATH
tok() { HIP_VISIBLE_DEVICES= "$bin/llama-tokenize" -m "$M" "$@" --ids --no-bos --log-disable 2>/dev/null | tr -d '[] ' | tail -1; }
cat "$repo"/docs/reference/*/*.md > "$B/swift-bench.text"
body=$(tok -f "$B/swift-bench.text")
head=$(tok -p '<|im_start|>user
Summarise the following documentation.

')
tail=$(tok -p '<|im_end|>
<|im_start|>assistant
')
for d in "$@"; do
  ids="$head,$(echo "$body" | cut -d, -f1-"$d"),$tail"
  echo "$ids" > "$B/swift-bench-d$d.ids"
  n=$(echo "$ids" | tr ',' '\n' | wc -l)
  for r in $(seq 1 ${REPS:-2}); do
    log="$B/swift-bench-d$d.$r.raw.log"
    HIP_VISIBLE_DEVICES=${GPU:-2} "$T/build_gfx1201/strata" --pack "$P" --native "$M" --ple-gguf "$M" --tokens-file "$B/swift-bench-d$d.ids" \
      --max-new ${MAX_NEW:-256} --max-context $((d + 2048)) --stats --spec 2 --prefill auto --expert-cache auto \
      --expert-profile "$T/data/expert-profile.bin" > "$log" 2>&1
    echo "d$d rep $r rc=$? prompt $n ids: $(grep -aE '^(decode|prefill) ' "$log" | tr -s ' ' | tr '\n' ';') $(grep -aoE 'drafts accepted [0-9]+ of [0-9]+ \([0-9.]+\)' "$log" | head -1)"
  done
done
echo SWIFT_BENCH_DONE
