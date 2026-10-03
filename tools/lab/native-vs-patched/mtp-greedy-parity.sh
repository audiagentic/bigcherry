#!/bin/bash
# Greedy parity under MTP: at temperature 0, speculative decoding must reproduce plain greedy output
# exactly. Runs <bin dir>/llama-server with and without draft-mtp (same flags, -sm tensor, dual XTX)
# on fixed prompts and diffs the generated text. A mismatch means a numerics/data bug in the
# speculative path (PGC09: adaptive lowers MTP acceptance although every host AR is f32).
# Usage: mtp-greedy-parity.sh <bin dir> <out-dir> [extra server args...]
set -u
bin=$1 out=$2; shift 2
mkdir -p "$out"
model=/mnt/data/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
run() {
  local name=$1; shift
  local port=$((45000 + RANDOM % 2000))
  HIP_VISIBLE_DEVICES=0,1 BIGCHERRY_PATCH_TRACE=1 "$bin/llama-server" -m "$model" -ngl 99 -sm tensor --fit off \
    --parallel 1 -c 8192 --flash-attn on "$@" --port $port > "$out/$name.log" 2>&1 &
  local pid=$!
  for _ in $(seq 150); do curl -sf http://127.0.0.1:$port/health >/dev/null && break; sleep 2; done
  python3 - "$port" "$out/$name.json" <<'PY'
import json, sys, urllib.request
port, path = sys.argv[1:3]
prompts = ["Write a long detailed story about a lighthouse keeper.", "The capital of France is",
           "Explain how a hash table works, step by step."]
res = []
for p in prompts:
    body = {"prompt": p, "n_predict": 256, "temperature": 0, "cache_prompt": False, "ignore_eos": True}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=900).read())
    res.append({"prompt": p, "content": r["content"], "draft_n": r["timings"].get("draft_n"), "accepted": r["timings"].get("draft_n_accepted")})
json.dump(res, open(path, "w"), indent=1)
PY
  kill -INT $pid; wait $pid
}
run plain "$@"
run mtp --spec-type draft-mtp --spec-draft-n-max 5 "$@"
python3 - "$out" <<'PY'
import json, sys
out = sys.argv[1]
a = json.load(open(f"{out}/plain.json")); b = json.load(open(f"{out}/mtp.json"))
for x, y in zip(a, b):
    same = x["content"] == y["content"]
    i = next((k for k, (c, d) in enumerate(zip(x["content"], y["content"])) if c != d), None) if not same else None
    print(("MATCH   " if same else f"DIFFER@{i} ") + repr(x["prompt"][:40]), "accepted", y["accepted"], "/", y["draft_n"])
PY
echo PARITY_DONE
