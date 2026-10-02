#!/bin/bash
# Qwen3.8-27B Q8_0 speculative-decoding probe on dual XTX (-sm tensor): MTP depth 5 (production) vs the
# DFlash2 block drafter (Q8_0 / Q4_K_M, block size 8), unsloth's standalone MTP sidecar (Q4_0) and the
# DSpark drafter (Anbeeld Q8_0), each with the draft on the XTXs or on the 6900 XT
# (-devd; HIP_VISIBLE_DEVICES=0,1,3 makes the 6900 ROCm2). The R9700 (vLLM) is not used.
# Same per-layout measurement as tools/lab/flash-next/layout-probe.sh: ~1665-token prompt, 3 x 128 tokens,
# greedy 64-token parity check against the plain (no draft) run.
# Usage: probe-27b.sh <llama-server> <out-dir> [layout-name regex]
set -u
bin=$1 out=$2 only=${3:-}
mkdir -p "$out"
model=/mnt/data/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
common=(-m "$model" -ngl 99 --fit off -c 8192 --flash-attn on --parallel 1 --threads 8 -sm tensor -ub 2048 -b 2048 -lv 4)
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
probe plain 0,1
probe mtp5 0,1 --spec-type draft-mtp --spec-draft-n-max 5 -ctkd q8_0 -ctvd q8_0
probe mtp5-d6900 0,1,3 -dev ROCm0,ROCm1 -devd ROCm2 --spec-type draft-mtp --spec-draft-n-max 5 -ctkd q8_0 -ctvd q8_0
probe dflash-q8-n7 0,1 -md /mnt/data/llm-models/qwen3.8-27b/gguf/dflash/Qwen3.8-27B-DFlash2-Q8_0.gguf --spec-type draft-dflash --spec-draft-n-max 7
probe dflash-q8-n7-d6900 0,1,3 -dev ROCm0,ROCm1 -devd ROCm2 -md /mnt/data/llm-models/qwen3.8-27b/gguf/dflash/Qwen3.8-27B-DFlash2-Q8_0.gguf --spec-type draft-dflash --spec-draft-n-max 7
probe dflash-q8-n4-d6900 0,1,3 -dev ROCm0,ROCm1 -devd ROCm2 -md /mnt/data/llm-models/qwen3.8-27b/gguf/dflash/Qwen3.8-27B-DFlash2-Q8_0.gguf --spec-type draft-dflash --spec-draft-n-max 4
probe dflash-q4-n7-d6900 0,1,3 -dev ROCm0,ROCm1 -devd ROCm2 -md /mnt/data/llm-models/qwen3.8-27b/gguf/dflash/Qwen3.8-27B-DFlash2-Q4_K_M.gguf --spec-type draft-dflash --spec-draft-n-max 7
probe mtpside5 0,1 -md /mnt/data/llm-models/qwen3.8-27b/gguf/unsloth-mtp/mtp-Qwen3.8-27B-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 5 -ctkd q8_0 -ctvd q8_0
probe mtpside5-d6900 0,1,3 -dev ROCm0,ROCm1 -devd ROCm2 -md /mnt/data/llm-models/qwen3.8-27b/gguf/unsloth-mtp/mtp-Qwen3.8-27B-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 5 -ctkd q8_0 -ctvd q8_0
probe dspark-q8-n7 0,1 -md /mnt/data/llm-models/qwen3.8-27b/gguf/dspark/Qwen3.8-27B-DSpark-Q8_0.gguf --spec-type draft-dspark --spec-draft-n-max 7
probe dspark-q8-n7-d6900 0,1,3 -dev ROCm0,ROCm1 -devd ROCm2 -md /mnt/data/llm-models/qwen3.8-27b/gguf/dspark/Qwen3.8-27B-DSpark-Q8_0.gguf --spec-type draft-dspark --spec-draft-n-max 7
probe mtp5-b 0,1 --spec-type draft-mtp --spec-draft-n-max 5 -ctkd q8_0 -ctvd q8_0
probe dflash-q8-n7-d6900-b 0,1,3 -dev ROCm0,ROCm1 -devd ROCm2 -md /mnt/data/llm-models/qwen3.8-27b/gguf/dflash/Qwen3.8-27B-DFlash2-Q8_0.gguf --spec-type draft-dflash --spec-draft-n-max 7
for n in mtp5 mtp5-d6900 mtpside5 mtpside5-d6900 dspark-q8-n7 dspark-q8-n7-d6900 dflash-q8-n7 dflash-q8-n7-d6900 dflash-q8-n4-d6900 dflash-q4-n7-d6900; do
  [ -f "$out/plain.greedy.txt" ] && [ -f "$out/$n.greedy.txt" ] && { cmp -s "$out/plain.greedy.txt" "$out/$n.greedy.txt" && echo "greedy plain == $n" || echo "greedy plain != $n"; }
done
echo PROBE_DONE
