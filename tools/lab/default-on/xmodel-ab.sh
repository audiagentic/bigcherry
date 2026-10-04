#!/bin/bash
# Default-on decision for generic runtime sets (QFP22 follow-up): same binary, flags off (A) vs a BIGCHERRY_FEATURES
# set on (B), ABBA at two depths, greedy text per run. Every BIGCHERRY_* / GGML_HIP_* variable is cleared first, so
# A is the build's defaults. Usage: xmodel-ab.sh <llama-server> <out-dir> <features> <model.gguf> <server args...>
set -u
for v in $(env | grep -oE "^(BIGCHERRY_[A-Z0-9_]+|GGML_HIP_[A-Z0-9_]+)"); do unset "$v"; done
bin=$1 out=$2 feat=$3 model=$4; shift 4
mkdir -p "$out"
port=$((44000 + RANDOM % 1000))
for depth in ${DEPTHS:-4096 16384}; do
  i=0
  for arm in A B B A; do
    i=$((i + 1))
    envs=; [ $arm = B ] && envs="BIGCHERRY_FEATURES=$feat"
    log=$out/d$depth.$i.$arm.log
    env $envs "$bin" -m "$model" -ngl 99 --fit off -c $((depth * 2 + 4096)) -fa on --parallel 1 --threads 8 \
        --port $port --host 127.0.0.1 "$@" > "$log" 2>&1 &
    pid=$!
    ok=0
    for _ in $(seq 300); do
      curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { ok=1; break; }
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    if [ $ok = 1 ]; then
      python3 - "$depth" "$port" "$out/d$depth.$i.$arm.txt" <<'PY'
import json, sys, urllib.request
depth, port, txt = int(sys.argv[1]), sys.argv[2], sys.argv[3]
words = ("The quick brown fox jumps over the lazy dog while the river keeps flowing past the old mill. " * 4000).split()
prompt = " ".join(words[: int(depth * 0.75)]) + "\n\nSummarise the text above in detail:"
req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(
    {"prompt": prompt, "n_predict": 256, "temperature": 0, "seed": 1, "cache_prompt": False}).encode(),
    headers={"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=1800))
open(txt, "w").write(r.get("content", ""))
t = r["timings"]
acc = t.get("draft_n_accepted"); dn = t.get("draft_n")
print(f"prefill {t['prompt_per_second']:.1f} t/s, decode {t['predicted_per_second']:.1f} t/s" + (f", accepted {acc}/{dn}" if dn else ""))
PY
    else
      echo "SERVER_FAILED"; grep -E " E |error|assert" "$log" | head -3
    fi | sed "s/^/d$depth $i $arm: /"
    grep -h "BIGCHERRY_FEATURES $feat" "$log" | head -1 | sed "s/^/d$depth $i $arm: /"
    kill -INT "$pid" 2>/dev/null
    for _ in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null
    sleep 3
  done
done
echo "greedy identity (md5 -> count):"
md5sum "$out"/*.txt | awk '{print $1}' | sort | uniq -c
