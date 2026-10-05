#!/bin/bash
# MET01 / 1336 (upstream #29943 copy callback): Flash-Next on ONE GPU with routed experts in host memory
# (--n-cpu-moe), so the scheduler's host-weight copy path is the one under test. Arms on one binary:
#   O = BIGCHERRY_MOE_COPY=0            observation-only control: the callback counts, every host weight is copied whole
#   S = default                         selective expert copy in the callback, dense shortcut at 90%
#   R = BIGCHERRY_MOE_COPY_DENSE_PCT=0  selective copy always (upstream behaviour, no dense shortcut)
# Each arm serves, in one process: a short greedy request, the same request again, a multi-ubatch prompt, and the
# short request a third time (second-request and workload-shift integrity). Reports per request md5 of the text,
# prefill and decode t/s, and the callback counters the binary prints at exit (BIGCHERRY_PATCH_TRACE).
# Usage: moe-copy-ab.sh <llama-server> <out-dir>      env: GPU (HIP index, default 2), NCMOE (41), CTX (16384),
#                                                         LONG_TOKENS (4096), N_PREDICT (128)
# Only HIP_VISIBLE_DEVICES selects the card: also setting ROCR_VISIBLE_DEVICES filters twice and leaves no device.
set -u
bin=$1 out=$2
mkdir -p "$out"
model=${BC_MODEL:-/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf}
gpu=${GPU:-2}
args=(-m "$model" -ngl 99 --n-cpu-moe ${NCMOE:-41} --fit off -c ${CTX:-16384} -ub 512 -b 2048 --flash-attn on --parallel 1
      --threads 16 -lv 4 -ot '^(per_layer_token_embd|token_embd)\.weight$=CPU' -ctk f16 -ctv f16)
run() {  # <arm> [VAR=value...]
  local arm=$1; shift
  local port=$((47000 + RANDOM % 2000)) log="$out/$arm.server.log"
  env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=$gpu BIGCHERRY_PATCH_TRACE=1 "$@" "$bin" "${args[@]}" --port "$port" > "$log" 2>&1 &
  local pid=$! ok=0
  for _ in $(seq 900); do
    curl -sf "http://127.0.0.1:$port/health" >/dev/null && { ok=1; break; }
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  if [ "$ok" != 1 ]; then echo "$arm: SERVER_FAILED"; grep -E " E |error|assert|abort" "$log" | tail -5; kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; return; fi
  # a run that silently fell back to the CPU is not a measurement of the copy path
  if grep -q "no usable GPU found" "$log"; then echo "$arm: SERVER_FAILED (no GPU: $(grep -m1 "failed to initialize" "$log" | cut -c1-120))"; kill "$pid"; wait "$pid" 2>/dev/null; return; fi
  echo "$arm: vram $(rocm-smi --showmeminfo vram 2>/dev/null | grep "GPU\[$gpu\].*Total Used" | awk '{print int($NF/1048576)" MiB"}')"
  python3 - "$port" "$out" "$arm" "${LONG_TOKENS:-4096}" "${N_PREDICT:-128}" <<'PY'
import hashlib, json, sys, urllib.request
port, out, arm, long_tokens, n_predict = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
corpus = open("/mnt/data/bigcherry-work/corpus/kld-docs.txt", errors="replace").read()
def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=3600).read())
short = "Write a Python function that merges two sorted lists into one sorted list, then explain its complexity.\n"
long = corpus[: 4 * long_tokens] + "\n\nIn summary, the text above"
for name, prompt, n in (("short1", short, n_predict), ("short2", short, n_predict), ("long", long, 64), ("short3", short, n_predict)):
    r = post({"prompt": prompt, "n_predict": n, "temperature": 0, "cache_prompt": False, "seed": 1})
    t = r["timings"]
    open(f"{out}/{arm}.{name}.txt", "w").write(r["content"])
    print(f"{arm} {name}: md5 {hashlib.md5(r['content'].encode()).hexdigest()[:12]} prompt {t['prompt_n']} tok "
          f"{t['prompt_per_second']:.1f} t/s, decode {t['predicted_n']} tok {t['predicted_per_second']:.2f} t/s", flush=True)
PY
  kill -INT "$pid"
  for _ in $(seq 120); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -0 "$pid" 2>/dev/null && { echo "$arm: shutdown hung, SIGKILL"; kill -9 "$pid"; }
  wait "$pid" 2>/dev/null
  echo "$arm: $(grep -o "BIGCHERRY_PATCH_HIT patch=1336.*" "$log" | tail -1)"
}
for shard in "${model%-00001-of-*}"-0000[12]-*.gguf; do cat "$shard" > /dev/null; done   # warm the page cache
run O BIGCHERRY_MOE_COPY=0
run S
run R BIGCHERRY_MOE_COPY_DENSE_PCT=0
run S2
echo "identity across arms (md5 -> files):"
md5sum "$out"/*.short1.txt "$out"/*.short2.txt "$out"/*.short3.txt "$out"/*.long.txt | awk '{print $1}' | sort | uniq -c
echo "per request:"; for r in short1 short2 short3 long; do echo "  $r: $(md5sum "$out"/*.$r.txt | awk '{print substr($1,1,12)}' | sort | uniq -c | tr '\n' ' ')"; done
echo MOE_COPY_AB_DONE
