#!/bin/bash
# Qwen3.8-Flash-Next (qwen4exp, 87 GiB IQ4_XS) load + layout probe (QFN01). For each layout:
# start llama-server (tensor split is the expected winner: all cards read weights at once;
# layer split runs one card at a time), record per-device VRAM, send one ~1000-token prompt with 128 generated
# tokens (cache off), record prompt/decode tokens/s and draft acceptance, stop.
# GPU order: 0,1 = 7900 XTX, 2 = R9700 (all CPU PCIe), 3 = 6900 XT (chipset PCIe, kept last).
# Usage: layout-probe.sh <llama-server> <out-dir>
set -u
bin=$1 out=$2
mkdir -p "$out"
model=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
ple=(-ot '^per_layer_token_embd\.weight$=CPU')
common=(-m "$model" -ngl 99 --fit off -c 8192 --flash-attn on --parallel 1 --threads 16 -lv 4 "${ple[@]}")
probe() {
  local name=$1 vis=$2; shift 2
  local port=$((43000 + RANDOM % 2000)) log="$out/$name.server.log"
  echo "== $name (devices $vis): $*"
  HIP_VISIBLE_DEVICES=$vis ROCR_VISIBLE_DEVICES=$vis "$bin" "${common[@]}" "$@" --port "$port" > "$log" 2>&1 &
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
    t = post({"prompt": text + "\n\nSummarise the above in detail:", "n_predict": 128, "cache_prompt": False, "temperature": 0})["timings"]
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
probe cpu3-tensor 0,1,2 -sm tensor -ts 3,3,2
probe cpu3-layer 0,1,2 -sm layer -ts 5,6,5
probe all4-tensor 0,1,2,3 -sm tensor -ts 15,15,10,8
probe cpu3-tensor-mtp3 0,1,2 -sm tensor -ts 3,3,2 --spec-type draft-mtp --spec-draft-n-max 3
probe cpu3-tensor-ngram-mtp3 0,1,2 -sm tensor -ts 3,3,2 --spec-type ngram-mod,draft-mtp --spec-draft-n-max 3
for n in cpu3-layer all4-tensor; do
  [ -f "$out/cpu3-tensor.greedy.txt" ] && [ -f "$out/$n.greedy.txt" ] &&     { cmp -s "$out/cpu3-tensor.greedy.txt" "$out/$n.greedy.txt" && echo "greedy cpu3-tensor == $n" || echo "greedy cpu3-tensor != $n"; }
done
echo PROBE_DONE
