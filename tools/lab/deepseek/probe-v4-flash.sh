#!/bin/bash
# DeepSeek-V4-Flash-0731 UD-IQ3_XXS (deepseek4: 43 layers, 256 experts / 6 active + 1 shared, 97 GiB of which
# 90 GiB routed experts) load + layout probe. The model does not fit in VRAM (96 GiB total over four cards),
# so routed experts of some layers go to CPU RAM (--n-cpu-moe N = the first N layers) and, in the 4-card
# layouts, a few layers' experts are pinned whole to the 6900 XT. Same measurement as layout-probe.sh.
# Usage: probe-v4-flash.sh <llama-server> <out-dir> [layout-name regex]
set -u
bin=$1 out=$2 only=${3:-}
mkdir -p "$out"
model=/mnt/data/llm-models/deepseek-v4-flash/gguf/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf
common=(-m "$model" -ngl 99 --fit off -c 32768 --flash-attn on --parallel 1 --threads 16 -lv 4)
probe() {
  local name=$1 vis=$2; shift 2
  [[ -n "$only" && ! "$name" =~ ^(${only})$ ]] && return
  local port=$((43000 + RANDOM % 2000)) log="$out/$name.server.log"
  echo "== $name (devices $vis): $*"
  HIP_VISIBLE_DEVICES=$vis ROCR_VISIBLE_DEVICES=$vis ${PROBE_TASKSET:-} "$bin" "${common[@]}" "$@" --port "$port" > "$log" 2>&1 &
  local pid=$!
  local ok=0
  for _ in $(seq 300); do
    curl -sf "http://127.0.0.1:$port/health" >/dev/null && { ok=1; break; }
    kill -0 "$pid" 2>/dev/null || break
    sleep 2
  done
  if [ "$ok" != 1 ]; then echo "$name: SERVER_FAILED"; tail -5 "$log"; kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; return; fi
  rocm-smi --showmeminfo vram 2>/dev/null | grep "Total Used" > "$out/$name.vram.txt"
  python3 - "$port" "$name" "$out" <<'PY'
import json, sys, urllib.request
port, name, out = sys.argv[1:4]
def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=900).read())
text = open("/mnt/data/bigcherry-work/corpus/kld-docs.txt", errors="replace").read()[:5000]
post({"prompt": "Hello", "n_predict": 8, "cache_prompt": False})
rows = []
for i in range(3):
    t = post({"prompt": text + "\n\nSummarise the above in detail:", "n_predict": 128, "cache_prompt": False, "temperature": 0, "ignore_eos": True})["timings"]
    rows.append({k: t.get(k) for k in ("prompt_n", "prompt_per_second", "predicted_n", "predicted_per_second", "draft_n", "draft_n_accepted")})
greedy = post({"prompt": "List the first ten prime numbers and explain why 1 is not prime.", "n_predict": 64, "cache_prompt": False, "temperature": 0, "seed": 1})
open(f"{out}/{name}.greedy.txt", "w").write(greedy["content"])
json.dump(rows, open(f"{out}/{name}.timings.json", "w"), indent=1)
pp = sum(r["prompt_per_second"] for r in rows) / 3; tg = sum(r["predicted_per_second"] for r in rows) / 3
acc = [r for r in rows if r.get("draft_n")]
a = sum(r["draft_n_accepted"] for r in acc) / max(1, sum(r["draft_n"] for r in acc)) if acc else None
print(f"{name}: prompt {rows[0]['prompt_n']} tok at {pp:.1f} t/s, decode {tg:.1f} t/s" + (f", draft acceptance {100*a:.1f}%" if a is not None else ""))
PY
  cat "$out/$name.vram.txt"
  kill -INT "$pid"; wait "$pid"
}
probe t3-cmoe16 0,1,2 -sm tensor -ts 3,3,4 --n-cpu-moe 16
probe t3-cmoe12 0,1,2 -sm tensor -ts 3,3,4 --n-cpu-moe 12
probe t3-cmoe10 0,1,2 -sm tensor -ts 3,3,4 --n-cpu-moe 10
probe l3-cmoe12 0,1,2 -sm layer -ts 3,3,4 --n-cpu-moe 12
probe t4-cmoe6-6900 0,1,2,3 -sm tensor -ts 3,3,4,0 --n-cpu-moe 6 -ot 'blk\.(3[7-9]|4[0-2])\.ffn_(gate|up|down)_exps\.weight=ROCm3'
probe l4-cmoe4 0,1,2,3 -sm layer -ts 3,3,4,2 --n-cpu-moe 4
echo PROBE_DONE
