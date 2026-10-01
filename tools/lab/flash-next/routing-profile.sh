#!/bin/bash
# QFN02 step 1: MoE routing profile for Qwen3.8-Flash-Next. Runs llama-eval-callback (built into the
# same tree as <llama-server>) with --tensor-filter 'ffn_moe_topk.*' and BIGCHERRY_DEBUG_FULL_TENSORS=1
# (patch 1279) on a mixed prose + code prompt, -sm tensor on XTX,XTX,R9700, and writes per-layer
# expert usage counts plus skew (share of selections covered by the hottest 10/25/50% of experts).
# Usage: routing-profile.sh <llama-server> <out-dir>
set -u
server=$1 out=$2; mkdir -p "$out"
tree=$(dirname "$(dirname "$server")")
ec=$tree/bin/llama-eval-callback
[ -x "$ec" ] || cmake --build "$tree" --target llama-eval-callback -j > "$out/build-eval-callback.log" 2>&1 \
  || { echo "eval-callback build failed"; tail -20 "$out/build-eval-callback.log"; exit 1; }
model=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
root=$(cd "$(dirname "$0")/../../.." && pwd)
{ head -c 6000 /mnt/data/bigcherry-work/corpus/kld-docs.txt; echo; echo "Now some code:"; head -c 6000 "$root/tools/bigcherry/patch/campaign/build.py"; } > "$out/prompt.txt"
HIP_VISIBLE_DEVICES=0,1,2 ROCR_VISIBLE_DEVICES=0,1,2 BIGCHERRY_DEBUG_FULL_TENSORS=1 \
  "$ec" -m "$model" -ngl 99 --fit off -sm tensor -ts 3,3,2 -c 8192 -b 4096 -ub 4096 --flash-attn on \
  -ot '^per_layer_token_embd\.weight$=CPU' --tensor-filter 'ffn_moe_topk.*' -f "$out/prompt.txt" \
  > "$out/dump.txt" 2> "$out/stderr.txt"
echo "EVAL_EXIT=$?"
python3 - "$out" <<'PY'
import collections, json, re, sys
out = sys.argv[1]
layer = None; counts = collections.defaultdict(collections.Counter)
head = re.compile(r"ffn_moe_topk-(\d+)")
for line in open(f"{out}/dump.txt", errors="replace"):
    m = head.search(line)
    if m and "=" in line:
        layer = int(m.group(1)); continue
    if layer is not None and "[" in line:
        for v in re.findall(r"-?\d+(?:\.\d+)?", line):
            counts[layer][int(float(v))] += 1
stats = {}
for l, c in sorted(counts.items()):
    tot = sum(c.values()); vals = sorted(c.values(), reverse=True) + [0] * (512 - len(c))
    cov = lambda f: round(sum(vals[: int(512 * f)]) / tot, 3)
    stats[l] = {"selections": tot, "experts_used": len(c), "top10pct": cov(0.10), "top25pct": cov(0.25), "top50pct": cov(0.50)}
json.dump({"per_layer": stats, "counts": {l: dict(c) for l, c in counts.items()}}, open(f"{out}/routing.json", "w"))
if stats:
    import statistics as st
    for k in ("experts_used", "top10pct", "top25pct", "top50pct"):
        print(k, "mean", round(st.mean(s[k] for s in stats.values()), 3), "min", min(s[k] for s in stats.values()), "max", max(s[k] for s in stats.values()))
    print("layers", len(stats), "selections/layer", stats[min(stats)]["selections"])
else:
    print("no ffn_moe_topk data parsed")
PY
echo ROUTING_DONE
