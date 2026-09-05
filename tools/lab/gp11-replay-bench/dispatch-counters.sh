#!/bin/bash
# GP11/HI160/HI162: what does the dispatch hot path ACTUALLY do on a real model?
#
# Every hit-rate number quoted in this project so far (99.8% L1) came from a
# synthetic repeated-shape sweep of 110,020 dispatches. That workload has a
# handful of distinct signatures by construction, so it cannot tell us whether
# the 8-slot thread-local L1 is adequate for a real 27B transformer with
# prefill and decode shapes, fused variants, and MTP draft+verify contexts.
#
# If real L1 hit rate is far below 99.8%, two things follow: the deferral
# opportunity in HI158 is larger than estimated, and every miss takes a GLOBAL
# std::mutex for the L2 map -- which would make the cache a contention point
# rather than an optimisation.
#
# This needs no rebuild: the counters are runtime-gated, so an existing binary
# reports them when GGML_HIP_DISPATCH_COUNTERS is set.
#
# It answers, in one run:
#   dispatch_entries / l1_hits / l1_misses    -> is 8 slots enough?
#   native_select calls vs dispatch_entries   -> size of the HI158 opportunity
#   hw_key_builds / sig_digest_builds         -> per-miss cost multiplier
#   replay exact/miss + cache loaded          -> was the cache consulted at all
#   final tuned vs native launches            -> did tuned kernels ACTUALLY run
#                                                (only in a build carrying HI160)
set -u
H=/mnt/vault/development/llmhosts/llamacpp
M=${MODEL:-/mnt/vault/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf}
CACHE_PATH=${CACHE_PATH:-$HOME/.cache/bigcherry/tune-campaigns/b4d3cf708425/dispatch.cache}
BENCH_MODEL=${BENCH_MODEL:-qwen27b}
DEVICES=${DEVICES:-0,1}
PORT=${PORT:-18400}
DIGEST=${DIGEST:?set DIGEST to the build digest to run}
USE_CACHE=${USE_CACHE:-yes}
OUT=${OUT:-$HOME/dispatch-counters.log}

bin=$(find ~/.cache/bigcherry/builds -path "*/$DIGEST/bin/llama-server" | head -1)
[ -x "$bin" ] || { echo "no binary for digest $DIGEST" >&2; exit 1; }

export ROCR_VISIBLE_DEVICES=$DEVICES HIP_VISIBLE_DEVICES=$DEVICES
export LD_LIBRARY_PATH=$(dirname "$bin")
export GGML_HIP_DISPATCH_COUNTERS=1
# WITHOUT THIS THE WHOLE DISPATCH LAYER IS OFF.
#
# ggml_hip_parse_mode() returns GGML_HIP_DISPATCH_MODE_NATIVE when
# GGML_HIP_DISPATCH_MODE is unset, and ggml_hip_dispatch_mul_mat returns false
# immediately in native mode. So a "replay" build with a cache path set but no
# mode set never loads the cache, never resolves a signature, never counts
# anything, and runs pure upstream code -- while looking, from the outside,
# exactly like a working replay arm. Every bench in this session before this
# line was added measured a switched-off dispatch layer.
export GGML_HIP_DISPATCH_MODE=${DISPATCH_MODE:-replay}
# THE evidence channel. llama-server installs a log callback that swallows the
# library's GGML_LOG_INFO lines, so the startup cache-load line and the
# shutdown counter/coverage reports NEVER appear on stdout or stderr -- which
# is why runs that looked silent were in fact working. The JSON coverage file
# is written from the same flush hook (anchored at ggml_backend_cuda_free) and
# is the only reliable way to see what the dispatch layer did.
export GGML_HIP_DISPATCH_COVERAGE=${COVERAGE_JSON:-/tmp/coverage.json}
rm -f "$GGML_HIP_DISPATCH_COVERAGE"
# Without this the /shutdown route is never registered and the counter report
# -- which is emitted at shutdown -- is lost to kill -9. See TEST.md.
export LLAMA_SERVER_ENABLE_SHUTDOWN=1
if [ "$USE_CACHE" = "yes" ]; then export GGML_HIP_DISPATCH_CACHE=$CACHE_PATH; else unset GGML_HIP_DISPATCH_CACHE; fi

: > "$OUT"
echo "start $(date -Is) digest=$DIGEST cache=$USE_CACHE bin=$bin" >> "$OUT"
rm -f /tmp/counters.log
nohup "$bin" -m "$M" --port $PORT --host 127.0.0.1 --parallel 1 --metrics \
  -sm tensor --fit off --spec-type draft-mtp --spec-draft-n-max 4 > /tmp/counters.log 2>&1 &
P=$!
ok=0; for i in $(seq 1 150); do curl -sf -o /dev/null http://127.0.0.1:$PORT/health && { ok=1; break; }; sleep 3; done
if [ $ok -eq 0 ]; then echo "SERVER_FAIL $(tail -3 /tmp/counters.log)" >> "$OUT"; kill -9 $P; exit 1; fi

grep -a "replay cache .* loaded" /tmp/counters.log >> "$OUT" || echo "no cache-load line" >> "$OUT"

(cd $H && timeout 3600 python3 bench/run_bench.py --bench-type server-bench \
   --server-url http://127.0.0.1:$PORT --model $BENCH_MODEL --bench-configs mtp-dual 2>&1 \
   | grep -aE "_tps:" | tr -s " ") >> "$OUT"

if ! curl -sf -X POST -o /dev/null "http://127.0.0.1:$PORT/shutdown" 2>/dev/null; then
  echo "WARN shutdown-endpoint-unavailable -- counters will be LOST" >> "$OUT"
fi
for i in $(seq 1 60); do kill -0 $P 2>/dev/null || break; sleep 1; done
kill -9 $P 2>/dev/null

echo "--- counters (log channel, usually empty: see note above) ---" >> "$OUT"
grep -aE "dispatch counters|native-force sites|dispatch coverage|replay v2|native-select timing" /tmp/counters.log >> "$OUT"
echo "--- coverage json (the real evidence) ---" >> "$OUT"
cat "$GGML_HIP_DISPATCH_COVERAGE" >> "$OUT" 2>/dev/null || echo "NO COVERAGE FILE -- backend teardown did not run" >> "$OUT"
echo "DISPATCH_COUNTERS_DONE" >> "$OUT"
