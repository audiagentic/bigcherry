#!/bin/bash
# MET01 / 1336 (upstream #29943 copy callback): Flash-Next on ONE GPU with routed experts in host memory
# (--n-cpu-moe), so the scheduler's host-weight copy path is the one under test. Arms on one binary:
#   O = BIGCHERRY_MOE_COPY=0            observation-only control: the callback counts, every host weight is copied whole
#   S = default                         selective expert copy in the callback, dense shortcut at 90%
#   R = BIGCHERRY_MOE_COPY_DENSE_PCT=0  selective copy always (upstream behaviour, no dense shortcut)
# Each arm serves, in one process: a short greedy request, the same request again, a multi-ubatch prompt, and the
# short request a third time (second-request and workload-shift integrity). Reports per request md5 of the text,
# prefill and decode t/s, and the callback counters the binary prints at exit (BIGCHERRY_PATCH_TRACE).
# ARMS=hop is the MET04 probe: every expert in VRAM, production tensor split (T) against a layer split over the two
# XTX with the routed experts of the last HOP_LAYERS layers on the R9700 (L28, L14). MTP=1 adds the MTP sidecar on
# the 6900 XT to every arm of any mode.
# ARMS=profile runs the 1338 lanes (frequency profile pinned in the cache; CACHE_MIB one size, PROFILE optional).
# ARMS=cache runs the 1337 expert-cache lanes instead: no cache, --moe-cache-mib for each of CACHE_MIB, no cache.
# Usage: moe-copy-ab.sh <llama-server> <out-dir>      env: GPU (HIP index, default 2), NCMOE (41), CTX (16384),
#                                                         LONG_TOKENS (4096), N_PREDICT (128)
# Only HIP_VISIBLE_DEVICES selects the card: also setting ROCR_VISIBLE_DEVICES filters twice and leaves no device.
set -u
bin=$1 out=$2
mkdir -p "$out"
model=${BC_MODEL:-/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf}
gpu=${GPU:-2}
base=(-m "$model" -ngl 99 --fit off -c ${CTX:-16384} -ub 512 -b 2048 --flash-attn on --parallel 1
      --threads 16 -lv 4 -ot '^(per_layer_token_embd|token_embd)\.weight$=CPU' -ctk f16 -ctv f16)
args=("${base[@]}" --n-cpu-moe ${NCMOE:-41})
# MTP sidecar on its own card (the 6900 XT), as in production: MTP=1 adds it to every arm
draft=${DRAFT:-/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf}
mtp=()
if [ "${MTP:-0}" = 1 ]; then
  # single-card arms: the target card and the 6900 XT are visible as ROCm0 and ROCm1; hop arms see all four
  if [ "${ARMS:-copy}" = hop ]; then DRAFT_DEV=3; else DRAFT_DEV=1; gpu=$gpu,3; args+=(-dev ROCm0); fi
  mtp=(-md "$draft" -devd ROCm$DRAFT_DEV --no-spec-draft-backend-sampling
       --spec-type draft-mtp --spec-draft-n-max ${SPEC_N:-3} -ctkd f16 -ctvd f16)
  export BIGCHERRY_DRAFT_VOCAB_N=65536
fi
if [ "${ARMS:-copy}" = hop ]; then   # MET04 probe: every expert in VRAM, dense layers on the two XTX
  gpu=0,1,2$([ "${MTP:-0}" = 1 ] && echo ,3)
  args=("${base[@]}")
fi
run() {  # <arm> [VAR=value...] [-- server args...]
  local arm=$1; shift
  local envs=() extra=()
  while [ $# -gt 0 ] && [ "$1" != -- ]; do envs+=("$1"); shift; done
  [ $# -gt 0 ] && { shift; extra=("$@"); }
  local port=$((47000 + RANDOM % 2000)) log="$out/$arm.server.log"
  env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=$gpu BIGCHERRY_PATCH_TRACE=1 "${envs[@]}" "$bin" "${args[@]}" "${mtp[@]}" "${extra[@]}" --port "$port" > "$log" 2>&1 &
  local pid=$! ok=0
  for _ in $(seq 900); do
    curl -sf "http://127.0.0.1:$port/health" >/dev/null && { ok=1; break; }
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  if [ "$ok" != 1 ]; then echo "$arm: SERVER_FAILED"; grep -E " E |error|assert|abort" "$log" | tail -5; kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; return; fi
  # a run that silently fell back to the CPU is not a measurement of the copy path
  if grep -q "no usable GPU found" "$log"; then echo "$arm: SERVER_FAILED (no GPU: $(grep -m1 "failed to initialize" "$log" | cut -c1-120))"; kill "$pid"; wait "$pid" 2>/dev/null; return; fi
  echo "$arm: vram $(rocm-smi --showmeminfo vram 2>/dev/null | grep "GPU\[[${gpu//,/}]\].*Total Used" | awk '{printf "%s%d", (NR>1?" / ":""), int($NF/1048576)} END {print " MiB (cards '"$gpu"')"}')"
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
  grep -oE "BIGCHERRY_PATCH_HIT patch=1338.*|llama_moe_cache: (ubatch|profile|wrote|layer).*" "$log" | sort -u | head -8 | sed "s/^/$arm: /"
  grep -iE "moe.?cache" "$log" | head -4 | cut -c1-200 | sed "s/^/$arm: /"
}
for shard in "${model%-00001-of-*}"-0000[12]-*.gguf; do cat "$shard" > /dev/null; done   # warm the page cache
if [ "${ARMS:-copy}" = hop ]; then
  # T  = production tensor split over the three cards (flashnext profile)
  # Ln = layer split of everything over the two XTX, with the routed experts of the LAST n layers on the R9700:
  #      each of those layers hands its activations XTX -> R9700 -> XTX through host memory (no P2P)
  exps() { echo "blk\\.($(seq -s'|' $((48 - $1)) 47))\\.ffn_(gate|up|down)_exps\\.weight=ROCm2"; }
  run T BIGCHERRY_FEATURES=flashnext BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0 -- -dev ROCm0,ROCm1,ROCm2 -sm tensor -ts 0.31,0.27,0.42
  for n in ${HOP_LAYERS:-28 14}; do
    # the layers that keep their experts (the first 48 - n) are ~1 GB each, the rest are light: give each XTX half
    # of the heavy ones (an even 24 / 24 layer split put all of them on the first card and ran out of memory)
    h=$(( (48 - n) / 2 ))
    run L$n BIGCHERRY_FEATURES=flashnext -- -dev ROCm0,ROCm1,ROCm2 -sm layer -ts $h,$((48 - h)),0 -ot "$(exps $n)"
  done
  run T2 BIGCHERRY_FEATURES=flashnext BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0 -- -dev ROCm0,ROCm1,ROCm2 -sm tensor -ts 0.31,0.27,0.42
elif [ "${ARMS:-copy}" = ub ]; then   # host-expert prefill against the micro-batch size: one expert upload per ubatch
  for ub in ${UB_LIST:-512 1024 2048 4096}; do run U$ub -- -ub $ub -b $ub; done
elif [ "${ARMS:-copy}" = record ]; then  # 1338: one run that writes $out/profile.bin from decode and prefill routing
  run W BIGCHERRY_MOE_CACHE_PROFILE_OUT=$out/profile.bin BIGCHERRY_MOE_CACHE_TRACE_OUT=$out/routing.bin BIGCHERRY_MOE_CACHE_LARGE=1 -- --moe-cache-mib ${CACHE_MIB:-22000}
elif [ "${ARMS:-copy}" = profile ]; then # 1338: W = LRU cache and write a profile, P = profile pinned + large batches
  mib=${CACHE_MIB:-22000}       # through the cache, PS = pinned, large batches off, W2 = LRU again. PROFILE = a profile
  prof=${PROFILE:-$out/profile.bin}   # from elsewhere (held out) instead of the one W writes
  run W BIGCHERRY_MOE_CACHE_PROFILE_OUT=$out/profile.bin -- --moe-cache-mib $mib
  run P BIGCHERRY_MOE_CACHE_PROFILE=$prof -- --moe-cache-mib $mib
  for pct in ${PIN_LIST:-}; do run P$pct BIGCHERRY_MOE_CACHE_PROFILE=$prof BIGCHERRY_MOE_CACHE_PIN_PCT=$pct -- --moe-cache-mib $mib; done   # pinned share sweep
  run PS BIGCHERRY_MOE_CACHE_PROFILE=$prof BIGCHERRY_MOE_CACHE_LARGE=0 -- --moe-cache-mib $mib
  run W2 -- --moe-cache-mib $mib
elif [ "${ARMS:-copy}" = cache ]; then # 1337: expert cache sizes at the same --n-cpu-moe (C0 = no cache, twice)
  run C0
  for mib in ${CACHE_MIB:-4096 2048 8192}; do run C$mib -- --moe-cache-mib $mib; done
  run C0b
else
  run O BIGCHERRY_MOE_COPY=0
  run S
  run R BIGCHERRY_MOE_COPY_DENSE_PCT=0
  run S2
fi
echo "identity across arms (md5 -> files):"
md5sum "$out"/*.short1.txt "$out"/*.short2.txt "$out"/*.short3.txt "$out"/*.long.txt | awk '{print $1}' | sort | uniq -c
echo "per request:"; for r in short1 short2 short3 long; do echo "  $r: $(md5sum "$out"/*.$r.txt | awk '{print substr($1,1,12)}' | sort | uniq -c | tr '\n' ' ')"; done
echo MOE_COPY_AB_DONE
