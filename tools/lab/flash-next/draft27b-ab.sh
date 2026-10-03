#!/bin/bash
# Qwen3.8-27B Q8_0 on the two XTX (-sm tensor, cpu-root AllReduce): built-in MTP head (drafts on the same TP GPUs,
# every draft token pays the tensor-split AllReduces) vs the Q4_0 MTP sidecar on the 6900 XT (-md ... -devd).
# Quick ABA per depth (A = built-in, B = sidecar), 256 greedy tokens after a ~DEPTH-token prompt; reports t/s,
# acceptance and ms/step. Not a full benchmark - a placement screen.
# Usage: draft27b-ab.sh <llama-server> <out-dir>
set -u
bin=$1 out=$2
mkdir -p "$out"
port=18741
M=/mnt/data/llm-models/qwen3.8-27b/gguf
model=$M/unsloth/Qwen3.8-27B-Q8_0.gguf
side=$M/unsloth-mtp/mtp-Qwen3.8-27B-Q4_0.gguf
COMMON="-m $model -ngl 99 -c 40960 -ub 512 -b 2048 -fa on --parallel 1 --threads 8 --port $port --host 127.0.0.1 --allreduce cpu-root -sm tensor -ts 1,1 --spec-type draft-mtp --spec-draft-n-max 4"

serve() {  # arm log
    local arm=$1 log=$2
    if [ "$arm" = builtin ]; then
        HIP_VISIBLE_DEVICES=0,1 "$bin" $COMMON > "$log" 2>&1 &
    else
        local dgpu=3  # sidecar = 6900 XT (physical GPU 3); sidecar_r9700 = R9700 (physical GPU 2)
        [ "$arm" = sidecar_r9700 ] && dgpu=2
        HIP_VISIBLE_DEVICES=0,1,$dgpu "$bin" $COMMON -dev ROCm0,ROCm1 -devd ROCm2 -md "$side" \
            --no-spec-draft-backend-sampling -ctkd f16 -ctvd f16 > "$log" 2>&1 &
    fi
    echo $!
}

request() {  # depth -> prints "tps acc/gen"
    python3 - "$1" "$port" <<'PY'
import json, sys, urllib.request
depth, port = int(sys.argv[1]), sys.argv[2]
words = ("The quick brown fox jumps over the lazy dog while the river keeps flowing past the old mill. " * 4000).split()
prompt = " ".join(words[: int(depth * 0.75)]) + "\n\nSummarise the text above in detail:"
req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(
    {"prompt": prompt, "n_predict": 256, "temperature": 0, "seed": 1, "cache_prompt": False}).encode(),
    headers={"Content-Type": "application/json"})
t = json.load(urllib.request.urlopen(req, timeout=1800))["timings"]
print(t["predicted_per_second"], t.get("draft_n_accepted"), t.get("draft_n"), t["predicted_n"])
PY
}

for depth in 10240 32768; do
    for arm in ${ARMS:-builtin sidecar builtin}; do
        log=$out/d$depth.$arm.$RANDOM.log
        pid=$(serve $arm "$log")
        ok=0
        for _ in $(seq 300); do
            curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { ok=1; break; }
            kill -0 "$pid" 2>/dev/null || break
            sleep 1
        done
        if [ $ok = 1 ]; then
            request $depth | python3 -c '
import sys
tps, acc, gen, n = sys.stdin.read().split()
tps, n = float(tps), int(n)
acc = int(acc) if acc not in ("None", "") else 0
steps = max(1, n - acc)
print(f"d'"$depth"' '"$arm"': {tps:.1f} t/s, accepted {acc}/{gen}, {n/tps/steps*1e3:.1f} ms/step")'
        else
            echo "d$depth $arm: SERVER_FAILED"; grep -E " E |error|assert" "$log" | head -3
        fi
        kill -INT "$pid" 2>/dev/null
        for _ in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
        kill -9 "$pid" 2>/dev/null
        wait "$pid" 2>/dev/null
    done
done
echo ALL_JOBS_DONE
