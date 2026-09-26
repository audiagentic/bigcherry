#!/bin/bash
# One-off pre-flight for the NRO07/NRO08 series-2 workload: does llama-server
# with --backend-sampling actually run 1256's TOP_K routes (ggml_top_k over the
# full vocabulary) under MTP decode? Starts one server from a 1256 subject
# build, sends one top_k=20 completion, and reports the route markers seen.
# Usage: topk_backend_sampling_check.sh <subject-bin-dir> <model.gguf> <hip-prefix> <device> [port]
# Run only between queue jobs (never alongside a measurement).
set -u
bin=$1; model=$2; hip=$3; dev=$4; port=${5:-18400}
root=$(cd "$(dirname "$0")/../../.." && pwd)
work=$("$root/tools/lab/plan-qualification/work-root.sh" "$root")
out=$work/runs/topk-backend-sampling-check-$(date +%Y%m%dT%H%M%S)
mkdir -p "$out"
export HIP_VISIBLE_DEVICES=$dev ROCM_PATH=$hip HIP_PATH=$hip BIGCHERRY_PATCH_TRACE=1
unset ROCR_VISIBLE_DEVICES
"$bin/llama-server" -m "$model" --port "$port" --parallel 1 --metrics -ngl 99 --fit off \
    --spec-type draft-mtp --spec-draft-n-max 4 --backend-sampling > "$out/server.log" 2>&1 &
pid=$!
trap 'kill $pid 2>/dev/null; wait $pid 2>/dev/null' EXIT
for _ in $(seq 1 180); do
    curl -sf "http://127.0.0.1:$port/health" > /dev/null && break
    kill -0 $pid 2>/dev/null || { echo "server exited early"; tail -20 "$out/server.log"; exit 1; }
    sleep 1
done
curl -sf "http://127.0.0.1:$port/completion" -H 'Content-Type: application/json' \
    -d '{"prompt":"Write a short paragraph about mountain weather.","n_predict":64,"temperature":1.0,"top_p":0.95,"top_k":20,"seed":7,"ignore_eos":true}' \
    > "$out/completion.json"
kill $pid; wait $pid 2>/dev/null; trap - EXIT
echo "log: $out/server.log"
echo "tokens_predicted: $(python3 -c "import json;print(json.load(open('$out/completion.json')).get('tokens_predicted'))")"
echo "route markers:"; grep -o 'BIGCHERRY_PATCH_HIT patch=125[67][^ ]* path=topk_[a-z_]*' "$out/server.log" | sort | uniq -c
grep -q 'path=topk_parallel_radix' "$out/server.log" && echo "RESULT: radix route exercised" || echo "RESULT: radix route NOT seen"
