#!/bin/bash
# PGC09 contract-lane check: start the session's control and subject llama-server with the MTP lane's
# exact arguments plus BIGCHERRY_PATCH_TRACE=1, log the AllReduce provider, and time one 512-token
# decode (30-token prompt) on each. Usage: pgc09-lane-check.sh <control bin dir> <subject bin dir> <out>
set -u
ctl=$1 sub=$2 out=$3
mkdir -p "$out"
model=/mnt/vault/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
for arm in control subject; do
  bin=$ctl; [ $arm = subject ] && bin=$sub
  port=$((44000 + RANDOM % 2000))
  HIP_VISIBLE_DEVICES=0,1 BIGCHERRY_PATCH_TRACE=1 "$bin/llama-server" -m "$model" -ngl 99 \
    --parallel 1 --metrics -sm tensor --fit off ${SPEC_ARGS---spec-type draft-mtp --spec-draft-n-max 4} ${EXTRA_ARGS:-} \
    --port $port > "$out/$arm.log" 2>&1 &
  pid=$!
  for _ in $(seq 150); do curl -sf http://127.0.0.1:$port/health >/dev/null && break; sleep 2; done
  for i in 1 2 3; do
    curl -s http://127.0.0.1:$port/completion -H 'Content-Type: application/json' \
      -d "{\"prompt\":\"Write a long detailed story about a lighthouse keeper.\",\"n_predict\":${N_PREDICT:-512},\"temperature\":0,\"cache_prompt\":false,\"ignore_eos\":true}" \
      | python3 -c "import json,sys;t=json.load(sys.stdin)['timings'];print('$arm', round(t['predicted_per_second'],2), t.get('draft_n_accepted'), t.get('draft_n'))"
  done
  grep -h "BIGCHERRY_PATCH_HIT patch=0860" "$out/$arm.log" | head -1
  kill -INT $pid; wait $pid
done
echo CHECK_DONE
