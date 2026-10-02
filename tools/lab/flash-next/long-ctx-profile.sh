#!/bin/bash
# Flash-Next 192K deployment profile (owner: long context is the usability target).
# Config: -ts 2,2,3 (R9700 weighted: KV follows the -ts head split), q8_0 KV, MTP3 draft on the 6900,
# --allreduce cpu-root, ub512. Two passes over the same server config:
#   1) unprofiled: prefill + decode timings at several context depths (8K/32K/96K prompt), memory breakdown;
#   2) rocprofv3 --kernel-trace --memory-copy-trace --stats on a 32K prompt + 128 decode, for the per-kernel
#      split of prefill and decode at depth (attention vs MoE vs AllReduce vs copies).
# Usage: long-ctx-profile.sh <llama-server> <out-dir>
set -u
bin=$1 out=$2
mkdir -p "$out"
model=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
draft=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q8_0-qsa4.gguf
args=(-m "$model" -ngl 99 --fit off -c 196608 --flash-attn on --parallel 1 --threads 16 -lv 4
      -ot '^per_layer_token_embd\.weight$=CPU' -dev ROCm0,ROCm1,ROCm2 -devd ROCm3 -sm tensor -ts 2,2,3
      -md "$draft" --no-spec-draft-backend-sampling --spec-type draft-mtp --spec-draft-n-max 3
      -ctk q8_0 -ctv q8_0 -ctkd q8_0 -ctvd q8_0 --allreduce cpu-root)
export HIP_VISIBLE_DEVICES=0,1,2,3 ROCR_VISIBLE_DEVICES=0,1,2,3
run_pass() {  # <name> <depths...>; server optionally wrapped by $WRAP
  local name=$1; shift
  local port=$((45000 + RANDOM % 2000)) log="$out/$name.server.log"
  ${WRAP:-} "$bin" "${args[@]}" --port "$port" > "$log" 2>&1 &
  local pid=$! ok=0
  for _ in $(seq 600); do
    curl -sf "http://127.0.0.1:$port/health" >/dev/null && { ok=1; break; }
    kill -0 "$pid" 2>/dev/null || break
    sleep 2
  done
  if [ "$ok" != 1 ]; then echo "$name: SERVER_FAILED"; tail -5 "$log"; kill "$pid" 2>/dev/null; wait "$pid"; return; fi
  rocm-smi --showmeminfo vram 2>/dev/null | grep "Total Used" > "$out/$name.vram.txt"
  python3 - "$port" "$name" "$out" "$@" <<'PY'
import json, sys, urllib.request
port, name, out = sys.argv[1:4]; depths = [int(d) for d in sys.argv[4:]]
def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=3600).read())
corpus = open("/mnt/data/bigcherry-work/corpus/kld-docs.txt", errors="replace").read()
post({"prompt": "Hello", "n_predict": 8, "cache_prompt": False})
rows = []
for d in depths:
    text = (corpus * (1 + 4 * d // max(1, len(corpus))))[: 4 * d]  # ~4 chars/token
    t = post({"prompt": text + "\n\nSummarise the above in detail:", "n_predict": 128, "cache_prompt": False,
              "temperature": 0, "ignore_eos": True})["timings"]
    r = {k: t.get(k) for k in ("prompt_n", "prompt_per_second", "predicted_n", "predicted_per_second", "draft_n", "draft_n_accepted")}
    rows.append(r)
    print(f"{name}: prompt {r['prompt_n']} tok at {r['prompt_per_second']:.1f} t/s, decode {r['predicted_per_second']:.1f} t/s, accepted {r['draft_n_accepted']}/{r['draft_n']}", flush=True)
json.dump(rows, open(f"{out}/{name}.timings.json", "w"), indent=1)
PY
  cat "$out/$name.vram.txt"
  kill -INT "$pid"; wait "$pid"
  grep -h "memory breakdown\|ROCm\|Host " "$log" | grep common_memory_breakdown_print | tail -6
}
run_pass plain 8192 32768 98304
WRAP="rocprofv3 --kernel-trace --memory-copy-trace --stats --output-format csv -d $out/rocprof --" run_pass profiled 32768
python3 - "$out/rocprof" <<'PY'
import csv, glob, sys, collections
for path in sorted(glob.glob(f"{sys.argv[1]}/**/*kernel_stats.csv", recursive=True))[:1]:
    rows = list(csv.DictReader(open(path)))
    total = sum(float(r["TotalDurationNs"]) for r in rows)
    print(f"kernel stats ({path}): total {total/1e6:.1f} ms")
    for r in sorted(rows, key=lambda r: -float(r["TotalDurationNs"]))[:25]:
        print(f"{100*float(r['TotalDurationNs'])/total:5.1f}% {int(r['Calls']):7d} {r['Name'][:110]}")
for path in sorted(glob.glob(f"{sys.argv[1]}/**/*memory_copy_stats.csv", recursive=True))[:1]:
    print(f"memory copy stats ({path}):")
    for r in csv.DictReader(open(path)):
        print(f"  {r['Name']}: calls {r['Calls']} total {float(r['TotalDurationNs'])/1e6:.1f} ms")
PY
