#!/bin/bash
# Flash-Next long-context fit sweep. At 192K with -ts 2,2,3 the KV cache is only ~3 GB total, but the R9700 is
# full (33.9/34.2 GB) while each XTX has ~4 GB free: rebalance -ts to reach 256K and/or ub1024.
# Per config: start server (q8_0 KV, MTP3 draft on the 6900, cpu-root), record per-GPU VRAM, time one 8K prompt
# + 128 decode, stop. Failures to load are reported, not fatal. Usage: long-ctx-fit.sh <llama-server> <out-dir>
set -u
bin=$1 out=$2
mkdir -p "$out"
model=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
draft=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q8_0-qsa4.gguf
export HIP_VISIBLE_DEVICES=0,1,2,3 ROCR_VISIBLE_DEVICES=0,1,2,3
fit() {  # <name> <ctx> <ts> <ub> [target KV type]
  local name=$1 ctx=$2 ts=$3 ub=$4 kv=${5:-q8_0}
  [[ -n "${FIT_ONLY:-}" && ! "$name" =~ ^(${FIT_ONLY})$ ]] && return
  local port=$((47000 + RANDOM % 2000)) log="$out/$name.server.log"
  "$bin" -m "$model" -ngl 99 --fit off -c "$ctx" --flash-attn on --parallel 1 --threads 16 -lv 4 \
    -ot '^per_layer_token_embd\.weight$=CPU' -dev ROCm0,ROCm1,ROCm2 -devd ROCm3 -sm tensor -ts "$ts" \
    -md "$draft" --no-spec-draft-backend-sampling --spec-type draft-mtp --spec-draft-n-max 3 \
    -ctk $kv -ctv $kv -ctkd q8_0 -ctvd q8_0 --allreduce cpu-root -ub "$ub" -b 2048 --port "$port" > "$log" 2>&1 &
  local pid=$! ok=0
  for _ in $(seq 300); do
    curl -sf "http://127.0.0.1:$port/health" >/dev/null && { ok=1; break; }
    kill -0 "$pid" 2>/dev/null || break
    sleep 2
  done
  if [ "$ok" != 1 ]; then echo "$name: LOAD_FAILED $(grep -m1 -iE 'out of memory|failed to allocate|error' "$log")"; kill "$pid" 2>/dev/null; wait "$pid"; return; fi
  echo "$name vram: $(rocm-smi --showmeminfo vram 2>/dev/null | grep 'Total Used' | awk '{printf "%.1f ", $NF/1e9}')GB"
  python3 - "$port" "$name" <<'PY'
import json, sys, urllib.request
port, name = sys.argv[1:3]
def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=1800).read())
corpus = open("/mnt/data/bigcherry-work/corpus/kld-docs.txt", errors="replace").read()
text = (corpus * (1 + 32768 // max(1, len(corpus))))[:32768]
post({"prompt": "Hello", "n_predict": 8, "cache_prompt": False})
t = post({"prompt": text + "\n\nSummarise the above in detail:", "n_predict": 128, "cache_prompt": False, "temperature": 0, "ignore_eos": True})["timings"]
print(f"{name}: prompt {t['prompt_n']} tok at {t['prompt_per_second']:.1f} t/s, decode {t['predicted_per_second']:.1f} t/s, accepted {t.get('draft_n_accepted')}/{t.get('draft_n')}", flush=True)
PY
  kill -INT "$pid"  # bounded: a rocprofv3-wrapped server hung 6.5 h after SIGINT on 2026-10-03
  for _ in $(seq 120); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -0 "$pid" 2>/dev/null && { echo "$name: shutdown hung, SIGKILL"; kill -9 "$pid"; }
  wait "$pid"
}
fit 192k-223-ub512   196608 2,2,3 512     # reference (current deployment)
fit 192k-334-ub1024  196608 3,3,4 1024
fit 256k-334-ub512   262144 3,3,4 512
fit 256k-556-ub512   262144 5,5,6 512
fit 256k-556-ub1024  262144 5,5,6 1024
fit 256k-223-ub512   262144 2,2,3 512
fit 256k-445-ub1024  262144 4,4,5 1024
# round 2 (round 1: every rebalanced -ts OOMed; compute buffers, not KV, are the limit)
fit 256k-223-ub256       262144 2,2,3 256
fit 256k-223-ub512-kvq4  262144 2,2,3 512 q4_0
fit 256k-223-ub256-kvq4  262144 2,2,3 256 q4_0
fit 192k-223-ub1024-kvq4 196608 2,2,3 1024 q4_0
