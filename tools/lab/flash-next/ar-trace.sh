#!/bin/bash
# QFN01/goal: Flash-Next AllReduce census on the 1277 trace build. One llama-server per arm with
# BIGCHERRY_AR_SIZE_TRACE, a ~1665-token prefill then a 64-token decode (cache off), and a histogram of
# (bytes, shape, type, provider) per phase plus calls per decoded token. Arms: no MTP and MTP3 (draft on
# the 6900), both -sm tensor -ts 4,4,3 ub1024 on GPU0-2.
# Usage: ar-trace.sh <llama-server> <out-dir>
set -u
bin=$1 out=$2; mkdir -p "$out"
model=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
md=(-md /mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q8_0-qsa4.gguf --no-spec-draft-backend-sampling --spec-type draft-mtp --spec-draft-n-max 3 -devd ROCm3)
common=(-m "$model" -ngl 99 --fit off -c 8192 --flash-attn on --parallel 1 --threads 16 -ot '^per_layer_token_embd\.weight$=CPU'
        -dev ROCm0,ROCm1,ROCm2 -sm tensor -ts 4,4,3 -ub 1024 -b 2048)
arm() {
  local name=$1; shift
  local port=$((45000 + RANDOM % 2000)) log="$out/$name.server.log"
  HIP_VISIBLE_DEVICES=0,1,2,3 ROCR_VISIBLE_DEVICES=0,1,2,3 BIGCHERRY_AR_SIZE_TRACE=2000000 \
    "$bin" "${common[@]}" "$@" --port "$port" > "$log" 2>&1 &
  local pid=$!
  for _ in $(seq 300); do curl -sf "http://127.0.0.1:$port/health" >/dev/null && break; kill -0 $pid 2>/dev/null || break; sleep 2; done
  python3 - "$port" "$out" "$name" <<'PY'
import json, sys, urllib.request
port, out, name = sys.argv[1:4]
def ntrace():
    return open(f"{out}/{name}.server.log", errors="replace").read().count("BIGCHERRY_AR_SIZE bytes=")
def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=900).read())
text = open("/mnt/data/bigcherry-work/corpus/kld-docs.txt", errors="replace").read()[:5000]
post({"prompt": "Hello", "n_predict": 4, "cache_prompt": False})
m = open(f"{out}/{name}.marks.txt", "w")
m.write("PREFILL_START\n"); m.flush()
r = post({"prompt": text, "n_predict": 1, "cache_prompt": False})
m.write(f"PREFILL_DONE prompt_n={r['timings']['prompt_n']}\n"); m.flush()
r = post({"prompt": "Count slowly:", "n_predict": 64, "cache_prompt": False, "temperature": 0, "ignore_eos": True})
m.write(f"DECODE_DONE predicted_n={r['timings']['predicted_n']} tg={r['timings']['predicted_per_second']:.1f}\n")
PY
  kill -INT $pid; wait $pid
  python3 - "$out" "$name" <<'PY'
import collections, re, sys
out, name = sys.argv[1:3]
rows = re.findall(r"BIGCHERRY_AR_SIZE bytes=(\d+) ne0=(-?\d+) ne1=(-?\d+) type=(\S+) provider=(\S+)", open(f"{out}/{name}.server.log", errors="replace").read())
hist = collections.Counter(rows)
with open(f"{out}/{name}.histogram.txt", "w") as f:
    for (b, n0, n1, t, p), c in sorted(hist.items(), key=lambda kv: -kv[1]):
        f.write(f"{c:9d}  bytes={b:>10} ne0={n0} ne1={n1} type={t} provider={p}\n")
by_p = collections.Counter(r[4] for r in rows)
print(name, "total calls", len(rows), "by provider", dict(by_p))
print(open(f"{out}/{name}.histogram.txt").read()[:2500])
print(open(f"{out}/{name}.marks.txt").read())
PY
}
arm nomtp
arm mtp3 "${md[@]}"
echo TRACE_DONE
