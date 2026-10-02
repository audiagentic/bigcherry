#!/bin/bash
# RRVP05 first Vulkan screening on Brutus: Qwen3.8-27B Q8_0 on the stock Vulkan build (RADV), same
# per-layout measurement as tools/lab/dflash/probe-27b.sh (ubatch 2048, ~1665-token prompt, greedy check
# against the first layout). Devices are selected with GGML_VK_VISIBLE_DEVICES (Vulkan order on Brutus:
# 0,1 = 7900 XTX, 2 = R9700, 3 = 6900 XT, same as HIP); the ICD with VK_DRIVER_FILES (RADV default).
# Usage: probe-27b-vk.sh <llama-server> <out-dir> [layout-name regex] [timed requests per layout, default 1]
set -u
bin=$1 out=$2 only=${3:-} reps=${4:-1}
mkdir -p "$out"
model=/mnt/data/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
common=(-m "$model" -ngl 99 --fit off -c 8192 --flash-attn on --parallel 1 --threads 8 -ub 2048 -b 2048 -lv 4)
probe() {
  local name=$1 vis=$2; shift 2
  [[ -n "$only" && ! "$name" =~ ^(${only})$ ]] && return
  local port=$((43000 + RANDOM % 2000)) log="$out/$name.server.log"
  echo "== $name (devices $vis): $*"
  GGML_VK_VISIBLE_DEVICES=$vis VK_DRIVER_FILES=${VK_ICD:-/usr/share/vulkan/icd.d/radeon_icd.json} ${PROBE_TASKSET:-} "$bin" "${common[@]}" "$@" --port "$port" > "$log" 2>&1 &
  local pid=$!
  local ok=0
  for _ in $(seq 300); do
    curl -sf "http://127.0.0.1:$port/health" >/dev/null && { ok=1; break; }
    kill -0 "$pid" 2>/dev/null || break
    sleep 2
  done
  if [ "$ok" != 1 ]; then echo "$name: SERVER_FAILED"; tail -5 "$log"; kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; return; fi
  rocm-smi --showmeminfo vram 2>/dev/null | grep "Total Used" > "$out/$name.vram.txt"
  python3 - "$port" "$name" "$out" "$reps" <<'PY'
import json, sys, urllib.request
port, name, out = sys.argv[1:4]
reps = int(sys.argv[4])
def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=900).read())
text = open("/mnt/data/bigcherry-work/corpus/kld-docs.txt", errors="replace").read()[:5000]
post({"prompt": "Hello", "n_predict": 8, "cache_prompt": False})
rows = []
for i in range(reps):
    t = post({"prompt": text + "\n\nSummarise the above in detail:", "n_predict": 128, "cache_prompt": False, "temperature": 0, "ignore_eos": True})["timings"]
    rows.append({k: t.get(k) for k in ("prompt_n", "prompt_per_second", "predicted_n", "predicted_per_second", "draft_n", "draft_n_accepted")})
greedy = post({"prompt": "List the first ten prime numbers and explain why 1 is not prime.", "n_predict": 64, "cache_prompt": False, "temperature": 0, "seed": 1})
open(f"{out}/{name}.greedy.txt", "w").write(greedy["content"])
json.dump(rows, open(f"{out}/{name}.timings.json", "w"), indent=1)
pp = sum(r["prompt_per_second"] for r in rows) / reps; tg = sum(r["predicted_per_second"] for r in rows) / reps
acc = [r for r in rows if r.get("draft_n")]
a = sum(r["draft_n_accepted"] for r in acc) / max(1, sum(r["draft_n"] for r in acc)) if acc else None
print(f"{name}: prompt {rows[0]['prompt_n']} tok at {pp:.1f} t/s, decode {tg:.1f} t/s" + (f", draft acceptance {100*a:.1f}%" if a is not None else ""))
PY
  cat "$out/$name.vram.txt"
  kill -INT "$pid"; wait "$pid"
}
probe vk-layer 0,1 -sm layer
probe vk-tensor 0,1 -sm tensor
probe vk-layer-mtp5 0,1 -sm layer --spec-type draft-mtp --spec-draft-n-max 5 -ctkd q8_0 -ctvd q8_0
probe vk-tensor-mtp5 0,1 -sm tensor --spec-type draft-mtp --spec-draft-n-max 5 -ctkd q8_0 -ctvd q8_0
probe vk-tensor-3card 0,1,2 -sm tensor -ts 3,3,2
probe vk-single-r9700 2 -sm none -ngl 99
for n in vk-tensor vk-layer-mtp5 vk-tensor-mtp5 vk-tensor-3card vk-single-r9700; do
  [ -f "$out/vk-layer.greedy.txt" ] && [ -f "$out/$n.greedy.txt" ] && { cmp -s "$out/vk-layer.greedy.txt" "$out/$n.greedy.txt" && echo "greedy vk-layer == $n" || echo "greedy vk-layer != $n"; }
done
echo PROBE_DONE
