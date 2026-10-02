#!/bin/bash
# Qwen3.8-27B Q8_0 speculative-decoding x deployment matrix (ubatch 2048, the 27B runtime setting).
# Deployments: dual XTX -sm tensor; 3-card XTX+XTX+R9700 -sm tensor -ts 3,3,2; dual XTX -sm layer.
# Drafts: built-in MTP (depth 3/4/5), unsloth MTP sidecar (Q4_0), DFlash2 (Q8_0/Q4_K_M, block 8),
# DSpark (Q8_0, block 7). DFlash/DSpark only run with a layer-split target: they reuse the target's
# output.weight, which a single-device draft cannot read from the tensor-split (meta) buffer.
# A draft on its own device uses -devd ROCm0 (one XTX) or ROCm3 (the 6900 XT); all four GPUs are visible
# and -dev picks the target cards (a non-contiguous HIP+ROCR visible list drops the 6900).
# Same per-layout measurement as tools/lab/flash-next/layout-probe.sh: ~1665-token prompt, 3 x 128 tokens,
# greedy 64-token parity check against the plain (no draft) run.
# Usage: probe-27b.sh <llama-server> <out-dir> [layout-name regex] [timed requests per layout, default 3]
#   Screening = 1 request (does it load, is greedy output identical, rough speed); detail = 3+.
set -u
bin=$1 out=$2 only=${3:-} reps=${4:-3}
mkdir -p "$out"
model=/mnt/data/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
common=(-m "$model" -ngl 99 --fit off -c 8192 --flash-attn on --parallel 1 --threads 8 -ub 2048 -b 2048 -lv 4)
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
probe plain 0,1,2,3 -dev ROCm0,ROCm1 -sm tensor
probe mtp5 0,1,2,3 -dev ROCm0,ROCm1 -sm tensor --spec-type draft-mtp --spec-draft-n-max 5 -ctkd q8_0 -ctvd q8_0
probe mtp4 0,1,2,3 -dev ROCm0,ROCm1 -sm tensor --spec-type draft-mtp --spec-draft-n-max 4 -ctkd q8_0 -ctvd q8_0
probe mtp3 0,1,2,3 -dev ROCm0,ROCm1 -sm tensor --spec-type draft-mtp --spec-draft-n-max 3 -ctkd q8_0 -ctvd q8_0
probe mtpside5-d6900 0,1,2,3 -dev ROCm0,ROCm1 -sm tensor -devd ROCm3 -md /mnt/data/llm-models/qwen3.8-27b/gguf/unsloth-mtp/mtp-Qwen3.8-27B-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 5 -ctkd q8_0 -ctvd q8_0
probe plain-3card 0,1,2,3 -dev ROCm0,ROCm1,ROCm2 -sm tensor -ts 3,3,2
probe mtp5-3card 0,1,2,3 -dev ROCm0,ROCm1,ROCm2 -sm tensor -ts 3,3,2 --spec-type draft-mtp --spec-draft-n-max 5 -ctkd q8_0 -ctvd q8_0
probe mtpside5-3card-d6900 0,1,2,3 -dev ROCm0,ROCm1,ROCm2 -sm tensor -ts 3,3,2 -devd ROCm3 -md /mnt/data/llm-models/qwen3.8-27b/gguf/unsloth-mtp/mtp-Qwen3.8-27B-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 5 -ctkd q8_0 -ctvd q8_0
probe plain-layer 0,1,2,3 -dev ROCm0,ROCm1 -sm layer
probe mtp5-layer 0,1,2,3 -dev ROCm0,ROCm1 -sm layer --spec-type draft-mtp --spec-draft-n-max 5 -ctkd q8_0 -ctvd q8_0
probe dflash-q8-n7-layer 0,1,2,3 -dev ROCm0,ROCm1 -sm layer -devd ROCm0 -md /mnt/data/llm-models/qwen3.8-27b/gguf/dflash/Qwen3.8-27B-DFlash2-Q8_0.gguf --spec-type draft-dflash --spec-draft-n-max 7
probe dflash-q8-n7-layer-d6900 0,1,2,3 -dev ROCm0,ROCm1 -sm layer -devd ROCm3 -md /mnt/data/llm-models/qwen3.8-27b/gguf/dflash/Qwen3.8-27B-DFlash2-Q8_0.gguf --spec-type draft-dflash --spec-draft-n-max 7
probe dflash-q4-n7-layer-d6900 0,1,2,3 -dev ROCm0,ROCm1 -sm layer -devd ROCm3 -md /mnt/data/llm-models/qwen3.8-27b/gguf/dflash/Qwen3.8-27B-DFlash2-Q4_K_M.gguf --spec-type draft-dflash --spec-draft-n-max 7
probe dflash-q8-n4-layer 0,1,2,3 -dev ROCm0,ROCm1 -sm layer -devd ROCm0 -md /mnt/data/llm-models/qwen3.8-27b/gguf/dflash/Qwen3.8-27B-DFlash2-Q8_0.gguf --spec-type draft-dflash --spec-draft-n-max 4
probe dspark-q8-n6-layer 0,1,2,3 -dev ROCm0,ROCm1 -sm layer -devd ROCm0 -md /mnt/data/llm-models/qwen3.8-27b/gguf/dspark/Qwen3.8-27B-DSpark-Q8_0.gguf --spec-type draft-dspark --spec-draft-n-max 6
probe dspark-q8-n6-layer-d6900 0,1,2,3 -dev ROCm0,ROCm1 -sm layer -devd ROCm3 -md /mnt/data/llm-models/qwen3.8-27b/gguf/dspark/Qwen3.8-27B-DSpark-Q8_0.gguf --spec-type draft-dspark --spec-draft-n-max 6
probe mtpside5-dR9700 0,1,2,3 -dev ROCm0,ROCm1 -sm tensor -devd ROCm2 -md /mnt/data/llm-models/qwen3.8-27b/gguf/unsloth-mtp/mtp-Qwen3.8-27B-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 5 -ctkd q8_0 -ctvd q8_0
probe dflash-q8-n7-layer-dR9700 0,1,2,3 -dev ROCm0,ROCm1 -sm layer -devd ROCm2 -md /mnt/data/llm-models/qwen3.8-27b/gguf/dflash/Qwen3.8-27B-DFlash2-Q8_0.gguf --spec-type draft-dflash --spec-draft-n-max 7
probe dspark-q8-n6-layer-dR9700 0,1,2,3 -dev ROCm0,ROCm1 -sm layer -devd ROCm2 -md /mnt/data/llm-models/qwen3.8-27b/gguf/dspark/Qwen3.8-27B-DSpark-Q8_0.gguf --spec-type draft-dspark --spec-draft-n-max 6
probe p1286-dflash-q8-n7-dR9700 0,1,2,3 -dev ROCm0,ROCm1 -sm tensor -devd ROCm2 -md /mnt/data/llm-models/qwen3.8-27b/gguf/dflash/Qwen3.8-27B-DFlash2-Q8_0.gguf --spec-type draft-dflash --spec-draft-n-max 7
probe p1286-dflash-q8-n7-d6900 0,1,2,3 -dev ROCm0,ROCm1 -sm tensor -devd ROCm3 -md /mnt/data/llm-models/qwen3.8-27b/gguf/dflash/Qwen3.8-27B-DFlash2-Q8_0.gguf --spec-type draft-dflash --spec-draft-n-max 7
probe p1286-dflash-q8-n7-dx0 0,1,2,3 -dev ROCm0,ROCm1 -sm tensor -devd ROCm0 -md /mnt/data/llm-models/qwen3.8-27b/gguf/dflash/Qwen3.8-27B-DFlash2-Q8_0.gguf --spec-type draft-dflash --spec-draft-n-max 7
probe p1286-dspark-q8-n6-dR9700 0,1,2,3 -dev ROCm0,ROCm1 -sm tensor -devd ROCm2 -md /mnt/data/llm-models/qwen3.8-27b/gguf/dspark/Qwen3.8-27B-DSpark-Q8_0.gguf --spec-type draft-dspark --spec-draft-n-max 6
probe p1286-dspark-q8-n6-d6900 0,1,2,3 -dev ROCm0,ROCm1 -sm tensor -devd ROCm3 -md /mnt/data/llm-models/qwen3.8-27b/gguf/dspark/Qwen3.8-27B-DSpark-Q8_0.gguf --spec-type draft-dspark --spec-draft-n-max 6
probe p1286-mtp5 0,1,2,3 -dev ROCm0,ROCm1 -sm tensor --spec-type draft-mtp --spec-draft-n-max 5 -ctkd q8_0 -ctvd q8_0
probe dt-mtp5-a 0,1,2,3 -dev ROCm0,ROCm1 -sm tensor --spec-type draft-mtp --spec-draft-n-max 5 -ctkd q8_0 -ctvd q8_0
probe dt-mtp5-3card-a 0,1,2,3 -dev ROCm0,ROCm1,ROCm2 -sm tensor -ts 3,3,2 --spec-type draft-mtp --spec-draft-n-max 5 -ctkd q8_0 -ctvd q8_0
probe dt-mtp4-3card 0,1,2,3 -dev ROCm0,ROCm1,ROCm2 -sm tensor -ts 3,3,2 --spec-type draft-mtp --spec-draft-n-max 4 -ctkd q8_0 -ctvd q8_0
probe dt-mtp6-3card 0,1,2,3 -dev ROCm0,ROCm1,ROCm2 -sm tensor -ts 3,3,2 --spec-type draft-mtp --spec-draft-n-max 6 -ctkd q8_0 -ctvd q8_0
probe dt-mtpside5-3card-d6900 0,1,2,3 -dev ROCm0,ROCm1,ROCm2 -sm tensor -ts 3,3,2 -devd ROCm3 -md /mnt/data/llm-models/qwen3.8-27b/gguf/unsloth-mtp/mtp-Qwen3.8-27B-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 5 -ctkd q8_0 -ctvd q8_0
probe dt-dflash-q8-n7-3card-d6900 0,1,2,3 -dev ROCm0,ROCm1,ROCm2 -sm tensor -ts 3,3,2 -devd ROCm3 -md /mnt/data/llm-models/qwen3.8-27b/gguf/dflash/Qwen3.8-27B-DFlash2-Q8_0.gguf --spec-type draft-dflash --spec-draft-n-max 7
probe dt-dflash-q8-n7-dR9700-a 0,1,2,3 -dev ROCm0,ROCm1 -sm tensor -devd ROCm2 -md /mnt/data/llm-models/qwen3.8-27b/gguf/dflash/Qwen3.8-27B-DFlash2-Q8_0.gguf --spec-type draft-dflash --spec-draft-n-max 7
probe dt-dflash-q8-n4-dR9700 0,1,2,3 -dev ROCm0,ROCm1 -sm tensor -devd ROCm2 -md /mnt/data/llm-models/qwen3.8-27b/gguf/dflash/Qwen3.8-27B-DFlash2-Q8_0.gguf --spec-type draft-dflash --spec-draft-n-max 4
probe dt-dflash-q8-n10-dR9700 0,1,2,3 -dev ROCm0,ROCm1 -sm tensor -devd ROCm2 -md /mnt/data/llm-models/qwen3.8-27b/gguf/dflash/Qwen3.8-27B-DFlash2-Q8_0.gguf --spec-type draft-dflash --spec-draft-n-max 10
probe dt-dflash-q4-n7-dR9700 0,1,2,3 -dev ROCm0,ROCm1 -sm tensor -devd ROCm2 -md /mnt/data/llm-models/qwen3.8-27b/gguf/dflash/Qwen3.8-27B-DFlash2-Q4_K_M.gguf --spec-type draft-dflash --spec-draft-n-max 7
probe dt-mtp5-3card-b 0,1,2,3 -dev ROCm0,ROCm1,ROCm2 -sm tensor -ts 3,3,2 --spec-type draft-mtp --spec-draft-n-max 5 -ctkd q8_0 -ctvd q8_0
probe dt-mtp5-b 0,1,2,3 -dev ROCm0,ROCm1 -sm tensor --spec-type draft-mtp --spec-draft-n-max 5 -ctkd q8_0 -ctvd q8_0
probe dt-dflash-q8-n7-dR9700-b 0,1,2,3 -dev ROCm0,ROCm1 -sm tensor -devd ROCm2 -md /mnt/data/llm-models/qwen3.8-27b/gguf/dflash/Qwen3.8-27B-DFlash2-Q8_0.gguf --spec-type draft-dflash --spec-draft-n-max 7
probe q4-dflash-n7-dR9700-a 0,1,2,3 -dev ROCm0,ROCm1 -sm tensor -devd ROCm2 -md /mnt/data/llm-models/qwen3.8-27b/gguf/dflash/Qwen3.8-27B-DFlash2-Q4_K_M.gguf --spec-type draft-dflash --spec-draft-n-max 7
probe q4-dflash8-n7-dR9700-a 0,1,2,3 -dev ROCm0,ROCm1 -sm tensor -devd ROCm2 -md /mnt/data/llm-models/qwen3.8-27b/gguf/dflash/Qwen3.8-27B-DFlash2-Q8_0.gguf --spec-type draft-dflash --spec-draft-n-max 7
probe q4-dflash-n4-dR9700 0,1,2,3 -dev ROCm0,ROCm1 -sm tensor -devd ROCm2 -md /mnt/data/llm-models/qwen3.8-27b/gguf/dflash/Qwen3.8-27B-DFlash2-Q4_K_M.gguf --spec-type draft-dflash --spec-draft-n-max 4
probe q4-dflash-n10-dR9700 0,1,2,3 -dev ROCm0,ROCm1 -sm tensor -devd ROCm2 -md /mnt/data/llm-models/qwen3.8-27b/gguf/dflash/Qwen3.8-27B-DFlash2-Q4_K_M.gguf --spec-type draft-dflash --spec-draft-n-max 10
probe q4-dflash-n7-d6900 0,1,2,3 -dev ROCm0,ROCm1 -sm tensor -devd ROCm3 -md /mnt/data/llm-models/qwen3.8-27b/gguf/dflash/Qwen3.8-27B-DFlash2-Q4_K_M.gguf --spec-type draft-dflash --spec-draft-n-max 7
probe q4-dflash-n7-3card-d6900 0,1,2,3 -dev ROCm0,ROCm1,ROCm2 -sm tensor -ts 3,3,2 -devd ROCm3 -md /mnt/data/llm-models/qwen3.8-27b/gguf/dflash/Qwen3.8-27B-DFlash2-Q4_K_M.gguf --spec-type draft-dflash --spec-draft-n-max 7
probe q4-dflash8-n7-dR9700-b 0,1,2,3 -dev ROCm0,ROCm1 -sm tensor -devd ROCm2 -md /mnt/data/llm-models/qwen3.8-27b/gguf/dflash/Qwen3.8-27B-DFlash2-Q8_0.gguf --spec-type draft-dflash --spec-draft-n-max 7
probe q4-dflash-n7-dR9700-b 0,1,2,3 -dev ROCm0,ROCm1 -sm tensor -devd ROCm2 -md /mnt/data/llm-models/qwen3.8-27b/gguf/dflash/Qwen3.8-27B-DFlash2-Q4_K_M.gguf --spec-type draft-dflash --spec-draft-n-max 7
probe single-r9700 0,1,2,3 -dev ROCm2 -sm none
probe single-r9700-mtp5 0,1,2,3 -dev ROCm2 -sm none --spec-type draft-mtp --spec-draft-n-max 5 -ctkd q8_0 -ctvd q8_0
probe mtp5-b 0,1,2,3 -dev ROCm0,ROCm1 -sm tensor --spec-type draft-mtp --spec-draft-n-max 5 -ctkd q8_0 -ctvd q8_0
for n in mtp5 mtp4 mtp3 mtpside5-d6900 plain-3card mtp5-3card mtpside5-3card-d6900 plain-layer mtp5-layer dflash-q8-n7-layer dflash-q8-n7-layer-d6900 dflash-q4-n7-layer-d6900 dflash-q8-n4-layer dspark-q8-n6-layer dspark-q8-n6-layer-d6900 mtpside5-dR9700 dflash-q8-n7-layer-dR9700 dspark-q8-n6-layer-dR9700 p1286-dflash-q8-n7-dR9700 p1286-dflash-q8-n7-d6900 p1286-dflash-q8-n7-dx0 p1286-dspark-q8-n6-dR9700 p1286-dspark-q8-n6-d6900 p1286-mtp5 dt-mtp5-a dt-mtp5-3card-a dt-mtp4-3card dt-mtp6-3card dt-mtpside5-3card-d6900 dt-dflash-q8-n7-3card-d6900 dt-dflash-q8-n7-dR9700-a dt-dflash-q8-n4-dR9700 dt-dflash-q8-n10-dR9700 dt-dflash-q4-n7-dR9700 dt-mtp5-3card-b dt-mtp5-b dt-dflash-q8-n7-dR9700-b q4-dflash-n7-dR9700-a q4-dflash8-n7-dR9700-a q4-dflash-n4-dR9700 q4-dflash-n10-dR9700 q4-dflash-n7-d6900 q4-dflash-n7-3card-d6900 q4-dflash8-n7-dR9700-b q4-dflash-n7-dR9700-b single-r9700 single-r9700-mtp5 mtp5-b; do
  [ -f "$out/plain.greedy.txt" ] && [ -f "$out/$n.greedy.txt" ] && { cmp -s "$out/plain.greedy.txt" "$out/$n.greedy.txt" && echo "greedy plain == $n" || echo "greedy plain != $n"; }
done
echo PROBE_DONE
