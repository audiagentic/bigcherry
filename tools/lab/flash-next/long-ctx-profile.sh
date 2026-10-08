#!/bin/bash
# Flash-Next 192K deployment profile (owner: long context is the usability target).
# Config: -ts 2,2,3 (R9700 weighted: KV follows the -ts head split), q8_0 KV, MTP3 draft on the 6900,
# --allreduce cpu-root, ub512. Two passes over the same server config:
#   1) unprofiled: prefill + decode timings at several context depths (8K/32K/96K prompt), memory breakdown;
#   2) rocprofv3 --kernel-trace --memory-copy-trace --stats on a 32K prompt + 128 decode, for the per-kernel
#      split of prefill and decode at depth (attention vs MoE vs AllReduce vs copies).
# Usage: long-ctx-profile.sh <llama-server> <out-dir> [full|decode|perf|timing|probes|apitrace|synctrace]
# Other drafters: SPEC_TYPE replaces the MTP default (draft-dspark, draft-dflash), SPEC_PMIN sets --spec-draft-p-min,
# DRAFT names the draft GGUF, SPEC_N the block length. Single-word values, so they pass through AB_ENV. A DFlash /
# DSpark draft with this tensor-split target needs patch 1286 in the build.
set -u
bin=$1 out=$2
mkdir -p "$out"
model=${BC_MODEL:-/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf}
AR_ARG=(--allreduce "${AR:-cpu-root}"); [ "${AR:-}" = none ] && AR_ARG=()  # AR=none: binaries without --allreduce (native llama.cpp)
draft=${DRAFT:-/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q8_0-qsa4.gguf}
spec=(--no-spec-draft-backend-sampling --spec-type draft-mtp)
[ -n "${SPEC_TYPE:-}" ] && spec=(--spec-type "$SPEC_TYPE")           # e.g. draft-dspark, draft-dflash
[ -n "${SPEC_PMIN:-}" ] && spec+=(--spec-draft-p-min "$SPEC_PMIN")  # 0 for a DSpark draft without a confidence head
args=(-m "$model" -ngl 99 --fit off -c ${CTX:-196608} -ub ${UB:-512} -b ${B:-2048} --flash-attn ${FA:-on} --parallel 1 --threads 16 -lv 4
      -ot '^per_layer_token_embd\.weight$=CPU' -dev ROCm0,ROCm1,ROCm2 -devd ROCm3 -sm tensor -ts ${TS:-2,2,3}
      -md "$draft" "${spec[@]}" --spec-draft-n-max ${SPEC_N:-3}
      -ctk ${CTK:-q8_0} -ctv ${CTV:-q8_0} -ctkd ${CTKD:-${CTK:-q8_0}} -ctvd ${CTVD:-${CTV:-q8_0}} "${AR_ARG[@]}")
if [ "${NO_MTP:-}" = 1 ]; then  # deterministic greedy reference: no draft, so no acceptance-dependent batch shapes
  args=(-m "$model" -ngl 99 --fit off -c ${CTX:-196608} -ub ${UB:-512} -b ${B:-2048} --flash-attn ${FA:-on} --parallel 1 --threads 16 -lv 4
        -ot '^per_layer_token_embd\.weight$=CPU' -dev ${DEVS:-ROCm0,ROCm1,ROCm2} -sm tensor -ts ${TS:-2,2,3}
        -ctk ${CTK:-q8_0} -ctv ${CTV:-q8_0} "${AR_ARG[@]}")
fi
[ -n "${EXTRA_OT:-}" ] && args+=(-ot "$EXTRA_OT")  # extra placement override for every target-model mode
if [ "${CPU_REF:-}" = 1 ]; then  # f32 CPU reference (no tensor split, no flash attention): slow, accuracy only
  args=(-m "$model" -ngl 0 --device none -c ${CTX:-16384} -ub ${UB:-512} -b 2048 --flash-attn off --parallel 1
        --threads 20 -lv 4 -ctk f16 -ctv f16)
fi
export HIP_VISIBLE_DEVICES=0,1,2,3 ROCR_VISIBLE_DEVICES=0,1,2,3
# A model evicted from the page cache (e.g. by a 27B run in between) loads through mmap at ~24 MB/s (17 minutes
# observed, 2026-10-05); a sequential read of the shards first is ~780 MB/s. Only shards under half resident: the
# model (94 GB) does not fit the host page cache (91 GB) whole, and a normal load is fine at ~60% resident.
for shard in "${model%-00001-of-*}"-*.gguf; do
  [ -f "$shard" ] || continue
  cached=$(python3 "$(cd "$(dirname "$0")" && pwd)/page-cache-fraction.py" "$shard" 2>/dev/null)
  if [ -n "$cached" ] && [ "$cached" -lt 50 ]; then
    echo "warming page cache: $(basename "$shard") (${cached}% resident)"
    cat "$shard" > /dev/null
  fi
done
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
  if [ "$ok" != 1 ]; then
    echo "$name: SERVER_FAILED"
    tail -5 "$log"
    local startup_shutdown=0
    if kill -0 "$pid" 2>/dev/null; then
      startup_shutdown=1
      kill -INT "$pid" 2>/dev/null
    fi
    wait "$pid"
    local server_rc=$? server_sig=none
    if [ "$server_rc" -gt 128 ]; then
      server_sig=$(kill -l $((server_rc - 128)) 2>/dev/null || echo $((server_rc - 128)))
    fi
    echo "$name: SERVER_EXIT status=$server_rc signal=$server_sig phase=startup shutdown_requested=$startup_shutdown"
    [ "$server_rc" -ne 0 ] && return "$server_rc"
    return 1
  fi
  rocm-smi --showmeminfo vram 2>/dev/null | grep "Total Used" > "$out/$name.vram.txt"
  SERVER_PID=$pid PERF_OUT=${PERF_OUT:-} CACHE=${CACHE:-} DECODE_N=${DECODE_N:-128} python3 - "$port" "$name" "$out" "$@" <<'PY'
import json, sys, urllib.request
port, name, out = sys.argv[1:4]; depths = [int(d) for d in sys.argv[4:]]
def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=3600).read())
corpus = open("/mnt/data/bigcherry-work/corpus/kld-docs.txt", errors="replace").read()
ASK = "\n\n" + __import__("os").environ.get("ASK", "Summarise the above in detail:")  # the request after the filled context
post({"prompt": "Hello", "n_predict": 8, "cache_prompt": False})
rows = []
for d in depths:
    text = (corpus * (1 + 4 * d // max(1, len(corpus))))[: 4 * d]  # ~4 chars/token
    import os
    cache = os.environ.get("CACHE") == "1"
    if os.environ.get("PROBES"):  # fidelity probes: natural-continuation next-token distributions over one cached fill
        fill = post({"prompt": text, "n_predict": 1, "cache_prompt": True, "temperature": 0})["timings"]
        print(f"{name}: fill prefill {fill['prompt_n']} tok at {fill['prompt_per_second']:.1f} t/s", flush=True)
        probes = []
        for i in range(int(os.environ["PROBES"])):
            off = (i * 104729) % max(1, len(corpus) - 2000)
            r = post({"prompt": text + "\n\n" + corpus[off:off + 600], "n_predict": 1, "cache_prompt": True,
                      "temperature": 0, "n_probs": 10})
            probes.append(r.get("completion_probabilities", [{}])[0])
        json.dump(probes, open(f"{out}/{name}.{d}.probes.json", "w"))
        print(f"{name}: {len(probes)} probes saved", flush=True)
        continue
    if cache:  # fill the KV cache first; the timed request then reuses it and only decodes
        fill = post({"prompt": text + ASK, "n_predict": 1, "cache_prompt": True})["timings"]
        print(f"{name}: fill prefill {fill['prompt_n']} tok at {fill['prompt_per_second']:.1f} t/s", flush=True)
    if os.environ.get("ARM_FILE"):  # sync-tracer.so starts counting once this file exists
        open(os.environ["ARM_FILE"], "w").close()
    perf = None
    if os.environ.get("PERF_OUT"):  # host-side sampling of the server during the timed decode only
        import subprocess
        perf = subprocess.Popen(["/usr/lib/linux-tools/6.8.0-142-generic/perf", "record", "-F", "499", "-g",
                                 "-p", os.environ["SERVER_PID"], "-o", os.environ["PERF_OUT"]],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t = post({"prompt": text + ASK, "n_predict": int(os.environ["DECODE_N"]),
              "cache_prompt": cache, "temperature": 0, "ignore_eos": True,
              **({"n_probs": 5} if os.environ.get("REPEAT") == "1" else {})})
    if os.environ.get("REPEAT") == "1":
        json.dump(t.get("completion_probabilities", [])[:int(os.environ.get("PROBS_KEEP", "8"))], open(f"{out}/{name}.{d}.probs.json", "w"))
    # temperature 0: the decoded text is the greedy output at this depth, compared across A/B arms
    open(f"{out}/{name}.{d}.greedy.txt", "w").write(t["content"])
    if os.environ.get("REPEAT") == "1":  # same server, same cached prefix: does decode alone diverge?
        t2 = post({"prompt": text + ASK, "n_predict": int(os.environ["DECODE_N"]),
                   "cache_prompt": cache, "temperature": 0, "ignore_eos": True, "n_probs": 5})
        open(f"{out}/{name}.{d}.r1.greedy.txt", "w").write(t2["content"])
        json.dump(t2.get("completion_probabilities", [])[:8], open(f"{out}/{name}.{d}.r1.probs.json", "w"))
    t = t["timings"]
    if perf is not None:
        import signal
        perf.send_signal(signal.SIGINT); perf.wait()
    r = {k: t.get(k) for k in ("prompt_n", "prompt_per_second", "predicted_n", "predicted_per_second", "draft_n", "draft_n_accepted")}
    rows.append(r)
    print(f"{name}: prompt {r['prompt_n']} tok at {r['prompt_per_second']:.1f} t/s, decode {r['predicted_per_second']:.1f} t/s, accepted {r['draft_n_accepted']}/{r['draft_n']}", flush=True)
json.dump(rows, open(f"{out}/{name}.timings.json", "w"), indent=1)
PY
  local client_rc=$?
  cat "$out/$name.vram.txt"
  local shutdown_requested=0
  if kill -0 "$pid" 2>/dev/null; then
    shutdown_requested=1
    kill -INT "$pid"  # bounded: a rocprofv3-wrapped server hung 6.5 h after SIGINT on 2026-10-03
    for _ in $(seq 120); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -0 "$pid" 2>/dev/null && { echo "$name: shutdown hung, SIGKILL"; kill -9 "$pid"; }
  fi
  wait "$pid"
  local server_rc=$? server_sig=none
  if [ "$server_rc" -gt 128 ]; then
    server_sig=$(kill -l $((server_rc - 128)) 2>/dev/null || echo $((server_rc - 128)))
  fi
  echo "$name: SERVER_EXIT status=$server_rc signal=$server_sig phase=run shutdown_requested=$shutdown_requested client_status=$client_rc"
  grep -h "memory breakdown\|ROCm\|Host " "$log" | grep common_memory_breakdown_print | tail -6
  if [ "$shutdown_requested" = 0 ] && [ "$server_rc" -ne 0 ]; then
    return "$server_rc"
  fi
  return "$client_rc"
}
mode=${3:-full}
if [ "$mode" = full ]; then
  run_pass plain 8192 32768 98304
  WRAP="rocprofv3 --kernel-trace --memory-copy-trace --stats --output-format csv -d $out/rocprof --" run_pass profiled 32768
elif [ "$mode" = synctrace ]; then  # RNX01: call sites of hipStreamSynchronize during decode
  here=$(cd "$(dirname "$0")" && pwd)
  gcc -O2 -shared -fPIC "$here/sync-tracer.c" -o "$out/sync-tracer.so" -ldl
  rm -f "$out/arm"
  ARM_FILE=$out/arm DECODE_N=256 CACHE=1 \
    WRAP="env LD_PRELOAD=$out/sync-tracer.so SYNC_TRACER_ARM=$out/arm SYNC_TRACER_OUT=$out/sync-sites.txt" \
    run_pass synctrace ${DEPTH:-8192}
  head -120 "$out/sync-sites.txt" 2>/dev/null || echo "no sync-sites.txt (server did not exit cleanly)"
  exit 0
elif [ "$mode" = apitrace ]; then  # RNX01: HIP API cost of decode (graph launches vs kernel launches, copies)
  DECODE_N=256 CACHE=1 WRAP="rocprofv3 --hip-runtime-trace --kernel-trace --memory-copy-trace --output-format csv -d $out/rocprof --" run_pass apitrace ${DEPTH:-8192}
  python3 - "$out/rocprof" "$out/apitrace.timings.json" <<'PY'
import csv, glob, json, sys, collections
t = json.load(open(sys.argv[2]))[-1]
n = t["predicted_n"]; win = n / t["predicted_per_second"] * 1e9
steps = max(1, n - (t.get("draft_n_accepted") or 0))
path = sorted(glob.glob(f"{sys.argv[1]}/**/*hip_api_trace.csv", recursive=True))[0]
rows = list(csv.DictReader(open(path)))
end = max(int(r["End_Timestamp"]) for r in rows)
agg = collections.defaultdict(lambda: [0, 0]); per_tid = collections.Counter()
for r in rows:
    s = int(r["Start_Timestamp"])
    if s < end - win: continue
    a = agg[r["Function"]]; a[0] += 1; a[1] += int(r["End_Timestamp"]) - s
    per_tid[r.get("Thread_Id", "?")] += int(r["End_Timestamp"]) - s
print(f"apitrace decode window {win/1e9:.1f} s, {n} tokens, {steps} steps")
for f, (c, ns) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:20]:
    print(f"  {f:40s} {c/steps:8.1f} calls/step {ns/1e6/steps:8.3f} ms/step  {ns/max(c,1)/1e3:7.2f} us/call")
for tid, ns in per_tid.most_common(6):
    print(f"  thread {tid}: {ns/1e6/steps:.2f} ms/step in HIP API ({100*ns/win:.0f}% of wall)")
PY
  exit 0
elif [ "$mode" = prefillprof ]; then  # QFP17: kernel trace + stats of one uncached prefill fill at DEPTH (8 decode tokens)
  DECODE_N=8 CACHE=0 WRAP="rocprofv3 --kernel-trace --stats --output-format csv -d $out/rocprof --" run_pass prefillprof ${DEPTH:-20480}
elif [ "$mode" = probes ]; then  # fidelity: PROBES next-token distributions after one cached fill at DEPTH
  PROBES=${PROBES:-24} CACHE=1 run_pass probes ${DEPTH:-24576}
  exit 0
elif [ "$mode" = timing ]; then  # unprofiled decode at ~80K cached context (A/B arm)
  DECODE_N=${DECODE_N:-512} CACHE=1 run_pass timing ${DEPTH:-65536}
  exit $?
elif [ "$mode" = perf ]; then  # host-side: where does the CPU spend decode at depth (GPUs ~75% idle)?
  DECODE_N=1024 CACHE=1 PERF_OUT=$out/decode.perf.data run_pass perfdecode ${DEPTH:-65536}
  p=/usr/lib/linux-tools/6.8.0-142-generic/perf
  $p report -i "$out/decode.perf.data" --no-children --sort comm --stdio 2>/dev/null | grep -E "^ +[0-9]" | head -15
  $p report -i "$out/decode.perf.data" --no-children --sort comm,dso,sym --stdio -g none 2>/dev/null | grep -E "^ +[0-9]" | head -50
  exit 0
else  # decode: cached ~96K-token prompt, then a long decode, so decode at depth dominates the trace
  DECODE_N=1024 CACHE=1 WRAP="rocprofv3 --kernel-trace --memory-copy-trace --stats --output-format csv -d $out/rocprof --" run_pass decode ${DEPTH:-65536}
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
