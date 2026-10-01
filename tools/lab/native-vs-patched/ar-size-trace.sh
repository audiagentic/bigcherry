#!/bin/bash
# One llama-server run on the adaptive-size-trace binary (27B Q8_0, dual XTX, plain decode, ubatch 512):
# trace the first N adaptive AllReduce calls, send a 1024-token prompt then a 64-token decode,
# and write a size/provider histogram. Usage: ar-size-trace.sh <llama-server> <out-dir>
set -u
bin=$1 out=$2
mkdir -p "$out"
model=/mnt/vault/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
port=$((40000 + RANDOM % 2000))
HIP_VISIBLE_DEVICES=0,1 ROCR_VISIBLE_DEVICES=0,1 BIGCHERRY_AR_SIZE_TRACE=200000 \
  "$bin" -m "$model" -sm tensor -ngl 99 --fit off -c 8192 --flash-attn on --ubatch-size 512 \
  --batch-size 2048 --threads 8 --parallel 1 --allreduce adaptive --port "$port" > "$out/server.log" 2>&1 &
pid=$!
for _ in $(seq 120); do curl -sf "http://127.0.0.1:$port/health" >/dev/null && break; sleep 2; done
python3 - "$port" "$out" <<'PY'
import json, sys, urllib.request
port, out = sys.argv[1], sys.argv[2]
def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())
words = ("the quick brown fox jumps over the lazy dog " * 200).split()
open(f"{out}/marks.txt", "a").write("PREFILL_START\n")
r = post({"prompt": " ".join(words[:1000]), "n_predict": 1, "cache_prompt": False})
open(f"{out}/marks.txt", "a").write(f"PREFILL_DONE prompt_n={r['timings']['prompt_n']}\n")
r = post({"prompt": "Count:", "n_predict": 64, "cache_prompt": False})
open(f"{out}/marks.txt", "a").write(f"DECODE_DONE predicted_n={r['timings']['predicted_n']}\n")
PY
kill -INT "$pid"; wait "$pid"
python3 - "$out" <<'PY'
import collections, re, sys
out = sys.argv[1]
rows = re.findall(r"BIGCHERRY_AR_SIZE bytes=(\d+) ne0=(-?\d+) ne1=(-?\d+) type=(\S+) provider=(\S+)", open(f"{out}/server.log").read())
hist = collections.Counter((int(b), int(n0), int(n1), t, p) for b, n0, n1, t, p in rows)
with open(f"{out}/histogram.txt", "w") as f:
    for (b, n0, n1, t, p), c in sorted(hist.items()):
        f.write(f"{c:8d}  bytes={b:>10}  ne0={n0} ne1={n1} type={t} provider={p}\n")
print(open(f"{out}/histogram.txt").read())
PY
echo TRACE_DONE
