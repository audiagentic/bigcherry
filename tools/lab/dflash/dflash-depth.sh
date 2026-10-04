#!/bin/bash
# QFP21: DFlash vs built-in MTP5 on Qwen3.8-27B Q8_0 dual-XTX tensor split at depth (1.6K only was tested):
# arms mtp5 | dflash2 Q4_K_M n6 on the R9700 | dflash2 Q8 n6 on the R9700; ABA rotation per depth (mtp5 first and last),
# prefill t/s (uncached prompt) and decode t/s, 256 greedy tokens, greedy text per run.
# Usage: dflash-depth.sh <llama-server with 1286> <out-dir>
set -u
for v in $(env | grep -oE "^(BIGCHERRY_[A-Z0-9_]+|GGML_HIP_[A-Z0-9_]+)"); do unset "$v"; done
bin=$1 out=$2
mkdir -p "$out"
M=/mnt/data/llm-models/qwen3.8-27b/gguf
port=$((47000 + RANDOM % 1000))
common=(-m $M/mtp/Qwen3.8-27B-Q8_0.gguf -ngl 99 --fit off -fa on --parallel 1 --threads 8 -ub 2048 -b 2048 -dev ROCm0,ROCm1 -sm tensor)
arm_args() {
  case $1 in
    mtp5)    echo "--spec-type draft-mtp --spec-draft-n-max 5 -ctkd q8_0 -ctvd q8_0" ;;
    dfq4n6)  echo "-devd ROCm2 -md $M/dflash/Qwen3.8-27B-DFlash2-Q4_K_M.gguf --spec-type draft-dflash --spec-draft-n-max 6" ;;
    dfq8n6)  echo "-devd ROCm2 -md $M/dflash/Qwen3.8-27B-DFlash2-Q8_0.gguf --spec-type draft-dflash --spec-draft-n-max 6" ;;
  esac
}
for depth in ${DEPTHS:-16384 65536}; do
  i=0
  for arm in mtp5 dfq4n6 dfq8n6 mtp5; do
    i=$((i + 1))
    log=$out/d$depth.$i.$arm.log
    HIP_VISIBLE_DEVICES=0,1,2,3 ROCR_VISIBLE_DEVICES=0,1,2,3 "$bin" "${common[@]}" -c $((depth * 2 + 4096)) $(arm_args $arm) \
        --port $port --host 127.0.0.1 > "$log" 2>&1 &
    pid=$!
    ok=0
    for _ in $(seq 400); do
      curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { ok=1; break; }
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    if [ $ok = 1 ]; then
      python3 - "$depth" "$port" "$out/d$depth.$i.$arm.txt" <<'PY' | sed "s/^/d$depth $i $arm: /"
import json, sys, urllib.request
depth, port, txt = int(sys.argv[1]), sys.argv[2], sys.argv[3]
words = ("The quick brown fox jumps over the lazy dog while the river keeps flowing past the old mill. " * 8000).split()
prompt = " ".join(words[: int(depth * 0.75)]) + "\n\nSummarise the text above in detail:"
req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(
    {"prompt": prompt, "n_predict": 256, "temperature": 0, "seed": 1, "cache_prompt": False}).encode(),
    headers={"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=3600))
open(txt, "w").write(r.get("content", ""))
t = r["timings"]
acc = t.get("draft_n_accepted"); dn = t.get("draft_n")
print(f"prefill {t['prompt_per_second']:.1f} t/s ({t['prompt_n']} tok), decode {t['predicted_per_second']:.1f} t/s"
      + (f", accepted {acc}/{dn}" if dn else ""))
PY
    else
      echo "d$depth $i $arm: SERVER_FAILED $(grep -hE ' E |error' "$log" | head -1 | cut -c1-150)"
    fi
    kill -INT "$pid" 2>/dev/null
    for _ in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null
    sleep 3
  done
done
echo "greedy identity per depth (md5 -> count):"
for depth in ${DEPTHS:-16384 65536}; do md5sum "$out"/d$depth.*.txt 2>/dev/null | awk '{print $1}' | sort | uniq -c | sed "s/^/d$depth /"; done
