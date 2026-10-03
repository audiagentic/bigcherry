#!/bin/bash
# Drafter comparison for Qwen3.8-27B on ONE GPU (XTX0), because DFlash2/DSpark cannot run against a tensor-split
# target (ggml_backend_sched_backend_id_from_cur abort: the drafter shares the target's embedding/lm_head; PNRO11).
# Target: UD-IQ4_XS (~15 GB). Arms: no drafter (greedy reference), built-in MTP (n=4), DFlash2 Q8_0, DFlash2 Q4_K_M,
# DSpark Q8_0 - all drafters on the same GPU. 256 greedy tokens after ~DEPTH-token prompts; reports t/s, acceptance,
# ms/step, and checks every arm's text against the reference (speculative decoding must be lossless).
# Usage: draft27b-single.sh <llama-server> <out-dir>
set -u
bin=$1 out=$2
mkdir -p "$out"
port=18761
G=/mnt/data/llm-models/qwen3.8-27b/gguf
COMMON="-m $G/unsloth/Qwen3.8-27B-UD-IQ4_XS.gguf -ngl 99 -c 40960 -ub 512 -b 2048 -fa on --parallel 1 --threads 8 --port $port --host 127.0.0.1"

serve() {
    local arm=$1 log=$2
    case $arm in
        nospec)    HIP_VISIBLE_DEVICES=0 "$bin" $COMMON > "$log" 2>&1 & ;;
        mtp)       HIP_VISIBLE_DEVICES=0 "$bin" $COMMON --spec-type draft-mtp --spec-draft-n-max 4 > "$log" 2>&1 & ;;
        dflash2_q8) HIP_VISIBLE_DEVICES=0 "$bin" $COMMON --spec-type draft-dflash -md $G/dflash/Qwen3.8-27B-DFlash2-Q8_0.gguf > "$log" 2>&1 & ;;
        dflash2_q4) HIP_VISIBLE_DEVICES=0 "$bin" $COMMON --spec-type draft-dflash -md $G/dflash/Qwen3.8-27B-DFlash2-Q4_K_M.gguf > "$log" 2>&1 & ;;
        dspark)    HIP_VISIBLE_DEVICES=0 "$bin" $COMMON --spec-type draft-dspark -md $G/dspark/Qwen3.8-27B-DSpark-Q8_0.gguf > "$log" 2>&1 & ;;
    esac
    echo $!
}

request() {  # depth txt-out
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
print(t["predicted_per_second"], t.get("draft_n_accepted"), t.get("draft_n"), t["predicted_n"])
PY
}

for depth in 4096 16384; do
    for arm in ${ARMS:-nospec mtp dflash2_q8 dflash2_q4 dspark mtp}; do
        log=$out/d$depth.$arm.$RANDOM.log
        pid=$(serve $arm "$log")
        ok=0
        for _ in $(seq 300); do
            curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { ok=1; break; }
            kill -0 "$pid" 2>/dev/null || break
            sleep 1
        done
        if [ $ok = 1 ]; then
            request $depth "$out/d$depth.$arm.txt" | python3 -c '
import sys
tps, acc, gen, n = sys.stdin.read().split()
tps, n = float(tps), int(n)
acc = int(acc) if acc not in ("None", "") else 0
print(f"d'"$depth"' '"$arm"': {tps:.1f} t/s, accepted {acc}/{gen}, {n/tps/max(1, n-acc)*1e3:.1f} ms/step")'
        else
            echo "d$depth $arm: SERVER_FAILED $(grep -m1 -E 'GGML_ASSERT| E |error' "$log" | cut -c1-160)"
        fi
        kill -INT "$pid" 2>/dev/null
        for _ in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
        kill -9 "$pid" 2>/dev/null
        wait "$pid" 2>/dev/null
    done
done
echo "== greedy identity vs no drafter"
for d in 4096 16384; do
    ref=$out/d$d.nospec.txt
    [ -f "$ref" ] || continue
    for f in $out/d$d.*.txt; do
        [ "$f" = "$ref" ] && continue
        cmp -s "$ref" "$f" && echo "d$d $(basename $f .txt | cut -d. -f2): IDENTICAL" || echo "d$d $(basename $f .txt | cut -d. -f2): DIFFERENT"
    done
done
echo ALL_JOBS_DONE
