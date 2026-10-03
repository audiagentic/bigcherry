#!/bin/bash
# Flash-Next 192K deployment profile (owner: long context is the usability target).
# Config: -ts 2,2,3 (R9700 weighted: KV follows the -ts head split), q8_0 KV, MTP3 draft on the 6900,
# --allreduce cpu-root, ub512. Two passes over the same server config:
#   1) unprofiled: prefill + decode timings at several context depths (8K/32K/96K prompt), memory breakdown;
#   2) rocprofv3 --kernel-trace --memory-copy-trace --stats on a 32K prompt + 128 decode, for the per-kernel
#      split of prefill and decode at depth (attention vs MoE vs AllReduce vs copies).
# Usage: long-ctx-profile.sh <llama-server> <out-dir> [full|decode|perf|timing]
set -u
bin=$1 out=$2
mkdir -p "$out"
model=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
draft=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q8_0-qsa4.gguf
args=(-m "$model" -ngl 99 --fit off -c 196608 --flash-attn on --parallel 1 --threads 16 -lv 4
      -ot '^per_layer_token_embd\.weight$=CPU' -dev ROCm0,ROCm1,ROCm2 -devd ROCm3 -sm tensor -ts 2,2,3
      -md "$draft" --no-spec-draft-backend-sampling --spec-type draft-mtp --spec-draft-n-max 3
      -ctk q8_0 -ctv q8_0 -ctkd q8_0 -ctvd q8_0 --allreduce cpu-root)
if [ "${NO_MTP:-}" = 1 ]; then  # deterministic greedy reference: no draft, so no acceptance-dependent batch shapes
  args=(-m "$model" -ngl 99 --fit off -c 196608 --flash-attn on --parallel 1 --threads 16 -lv 4
        -ot '^per_layer_token_embd\.weight$=CPU' -dev ROCm0,ROCm1,ROCm2 -sm tensor -ts 2,2,3
        -ctk q8_0 -ctv q8_0 --allreduce cpu-root)
fi
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
  SERVER_PID=$pid PERF_OUT=${PERF_OUT:-} CACHE=${CACHE:-} DECODE_N=${DECODE_N:-128} python3 - "$port" "$name" "$out" "$@" <<'PY'
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
    import os
    cache = os.environ.get("CACHE") == "1"
    if cache:  # fill the KV cache first; the timed request then reuses it and only decodes
        post({"prompt": text + "\n\nSummarise the above in detail:", "n_predict": 1, "cache_prompt": True})
    perf = None
    if os.environ.get("PERF_OUT"):  # host-side sampling of the server during the timed decode only
        import subprocess
        perf = subprocess.Popen(["/usr/lib/linux-tools/6.8.0-142-generic/perf", "record", "-F", "499", "-g",
                                 "-p", os.environ["SERVER_PID"], "-o", os.environ["PERF_OUT"]],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t = post({"prompt": text + "\n\nSummarise the above in detail:", "n_predict": int(os.environ["DECODE_N"]),
              "cache_prompt": cache, "temperature": 0, "ignore_eos": True})
    # temperature 0: the decoded text is the greedy output at this depth, compared across A/B arms
    open(f"{out}/{name}.{d}.greedy.txt", "w").write(t["content"])
    t = t["timings"]
    if perf is not None:
        import signal
        perf.send_signal(signal.SIGINT); perf.wait()
    r = {k: t.get(k) for k in ("prompt_n", "prompt_per_second", "predicted_n", "predicted_per_second", "draft_n", "draft_n_accepted")}
    rows.append(r)
    print(f"{name}: prompt {r['prompt_n']} tok at {r['prompt_per_second']:.1f} t/s, decode {r['predicted_per_second']:.1f} t/s, accepted {r['draft_n_accepted']}/{r['draft_n']}", flush=True)
json.dump(rows, open(f"{out}/{name}.timings.json", "w"), indent=1)
PY
  cat "$out/$name.vram.txt"
  kill -INT "$pid"  # bounded: a rocprofv3-wrapped server hung 6.5 h after SIGINT on 2026-10-03
  for _ in $(seq 120); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -0 "$pid" 2>/dev/null && { echo "$name: shutdown hung, SIGKILL"; kill -9 "$pid"; }
  wait "$pid"
  grep -h "memory breakdown\|ROCm\|Host " "$log" | grep common_memory_breakdown_print | tail -6
}
mode=${3:-full}
if [ "$mode" = full ]; then
  run_pass plain 8192 32768 98304
  WRAP="rocprofv3 --kernel-trace --memory-copy-trace --stats --output-format csv -d $out/rocprof --" run_pass profiled 32768
elif [ "$mode" = timing ]; then  # unprofiled decode at ~80K cached context (A/B arm)
  DECODE_N=512 CACHE=1 run_pass timing ${DEPTH:-65536}
  exit 0
elif [ "$mode" = perf ]; then  # host-side: where does the CPU spend decode at depth (GPUs ~75% idle)?
  DECODE_N=1024 CACHE=1 PERF_OUT=$out/decode.perf.data run_pass perfdecode ${DEPTH:-65536}
  p=/usr/lib/linux-tools/6.8.0-142-generic/perf
  $p report -i "$out/decode.perf.data" --no-children --sort comm --stdio 2>/dev/null | grep -E "^ +[0-9]" | head -15
  $p report -i "$out/decode.perf.data" --no-children --sort comm,dso,sym --stdio -g none 2>/dev/null | grep -E "^ +[0-9]" | head -50
  exit 0
else  # decode: cached ~96K-token prompt, then a long decode, so decode at depth dominates the trace
  DECODE_N=1024 CACHE=1 WRAP="rocprofv3 --kernel-trace --memory-copy-trace --stats --output-format csv -d $out/rocprof --" run_pass decode 65536
fi
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
python3 - "$out/rocprof" <<'PY'
# Where do the large device-to-device copies come from? Group by size and agent pair.
import csv, glob, sys, collections
for path in sorted(glob.glob(f"{sys.argv[1]}/**/*memory_copy_trace.csv", recursive=True))[:1]:
    groups = collections.defaultdict(lambda: [0, 0.0])
    for r in csv.DictReader(open(path)):
        if r["Direction"].endswith("DEVICE_TO_DEVICE"):  # this rocprofv3 reports no byte count
            k = (r["Source_Agent_Id"], r["Destination_Agent_Id"], "stream " + r["Stream_Id"])
            g = groups[k]; g[0] += 1; g[1] += (int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e6
    print("device-to-device copies by (src, dst, stream): calls, total ms")
    for k, (n, ms) in sorted(groups.items(), key=lambda kv: -kv[1][1])[:15]:
        print(f"  {k}: {n} calls, {ms:.1f} ms")
PY
[ "$mode" = decode ] && python3 - "$out/rocprof" "$out/decode.timings.json" <<'PY'
# Decode-window kernel table: rocprof traces the whole process (including the cache-fill prefill), so keep only
# kernels in the final predicted_n / predicted_per_second seconds of the trace.
import csv, glob, json, sys, collections
t = json.load(open(sys.argv[2]))[-1]
win_ns = t["predicted_n"] / t["predicted_per_second"] * 1e9
path = sorted(glob.glob(f"{sys.argv[1]}/**/*kernel_trace.csv", recursive=True))[0]
rows = list(csv.DictReader(open(path)))
end = max(int(r["End_Timestamp"]) for r in rows)
agg = collections.defaultdict(lambda: [0, 0])
for r in rows:
    if int(r["Start_Timestamp"]) >= end - win_ns:
        a = agg[r["Kernel_Name"]]; a[0] += 1; a[1] += int(r["End_Timestamp"]) - int(r["Start_Timestamp"])
total = sum(v[1] for v in agg.values())
print(f"decode window {win_ns/1e9:.1f} s ({t['predicted_n']} tokens): kernel total {total/1e6:.1f} ms")
for k, (n, ns) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:25]:
    print(f"{100*ns/total:5.1f}% {n:7d} {ns/1e6/t['predicted_n']:7.3f} ms/tok  {k[:100]}")
PY
