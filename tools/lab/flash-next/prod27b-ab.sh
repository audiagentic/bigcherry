#!/bin/bash
# QFP18 cross-model no-regression check: Qwen3.8-27B Q8_0 dual-XTX production config (-sm tensor, built-in MTP,
# cpu-root AllReduce) on two binaries, ABBA per depth (A = before, B = after). Reports prefill and decode t/s,
# acceptance, ms/step, and saves the greedy completion text per run (identity check: md5 of the .txt files).
# Usage: prod27b-ab.sh <before llama-server> <after llama-server> <out-dir> [after-arm env...]
# AR_FLAG (e.g. "--allreduce cpu-root") is passed to both arms; unset = each build's default all-reduce (the base
# build has no cpu-root provider: that comes from 1291).
set -u
A=$1 B=$2 out=$3 benv="${*:4}"
mkdir -p "$out"
port=18743
model=/mnt/data/llm-models/qwen3.8-27b/gguf/unsloth/Qwen3.8-27B-Q8_0.gguf
COMMON="-m $model -ngl 99 -c 40960 -ub 512 -b 2048 -fa on --parallel 1 --threads 8 --port $port --host 127.0.0.1 ${AR_FLAG:-} -sm tensor -ts 1,1 --spec-type draft-mtp --spec-draft-n-max 4"

request() {  # depth text-out -> "decode_tps prompt_tps acc gen n"
    python3 - "$1" "$port" "$2" <<'PY'
import json, sys, urllib.request
depth, port, txt_out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
words = ("The quick brown fox jumps over the lazy dog while the river keeps flowing past the old mill. " * 4000).split()
prompt = " ".join(words[: int(depth * 0.75)]) + "\n\nSummarise the text above in detail:"
req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(
    {"prompt": prompt, "n_predict": 256, "temperature": 0, "seed": 1, "cache_prompt": False}).encode(),
    headers={"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=1800))
open(txt_out, "w").write(r.get("content", ""))
t = r["timings"]
print(t["predicted_per_second"], t["prompt_per_second"], t.get("draft_n_accepted"), t.get("draft_n"), t["predicted_n"])
PY
}

for depth in 10240 32768; do
    i=0
    for arm in A B B A; do
        i=$((i + 1))
        bin=$A envs=; [ $arm = B ] && { bin=$B; envs=$benv; }
        log=$out/d$depth.$i.$arm.log
        env $envs HIP_VISIBLE_DEVICES=0,1 "$bin" $COMMON > "$log" 2>&1 &
        pid=$!
        ok=0
        for _ in $(seq 300); do
            curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { ok=1; break; }
            kill -0 "$pid" 2>/dev/null || break
            sleep 1
        done
        if [ $ok = 1 ]; then
            request $depth "$out/d$depth.$i.$arm.txt" | python3 -c '
import sys
d, p, acc, gen, n = sys.stdin.read().split()
d, p, n = float(d), float(p), int(n)
acc = int(acc) if acc not in ("None", "") else 0
print(f"d'"$depth"' '"$i"' '"$arm"': prefill {p:.1f} t/s, decode {d:.1f} t/s, accepted {acc}/{gen}, {n/d/max(1, n-acc)*1e3:.1f} ms/step")'
        else
            echo "d$depth $i $arm: SERVER_FAILED"; grep -E " E |error|assert" "$log" | head -3
        fi
        kill -INT "$pid" 2>/dev/null
        for _ in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
        kill -9 "$pid" 2>/dev/null
        sleep 5
    done
done
echo "greedy identity (md5 -> count):"
md5sum "$out"/*.txt | awk '{print $1}' | sort | uniq -c
