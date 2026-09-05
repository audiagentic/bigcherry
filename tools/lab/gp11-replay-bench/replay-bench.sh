#!/bin/bash
# GP11/HI158: replay (tuned winners) vs control (none) vs llama-native,
# end-to-end on 27B / dual gfx1100 / MTP. See README.md.
set -u
H=/mnt/vault/development/llmhosts/llamacpp
M=/mnt/vault/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
CACHE=~/.cache/bigcherry/tune-campaigns/b4d3cf708425/dispatch.cache
# Not 8080: llama-swap owns that in production on this host.
PORT=18400
cell () {
  local name=$1 bin=$2 usecache=$3
  export ROCR_VISIBLE_DEVICES=0,1 HIP_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH=$(dirname $bin)
  if [ "$usecache" = "yes" ]; then export GGML_HIP_DISPATCH_CACHE=$CACHE; else unset GGML_HIP_DISPATCH_CACHE; fi
  rm -f /tmp/r.log
  nohup "$bin" -m "$M" --port $PORT --host 127.0.0.1 --parallel 1 --metrics \
    -sm tensor --fit off --spec-type draft-mtp --spec-draft-n-max 4 > /tmp/r.log 2>&1 &
  local P=$!
  local ok=0; for i in $(seq 1 150); do curl -sf -o /dev/null http://127.0.0.1:$PORT/health && { ok=1; break; }; sleep 3; done
  if [ $ok -eq 0 ]; then echo "$name | SERVER_FAIL: $(tail -2 /tmp/r.log|tr "\n" " ")"; kill -9 $P 2>/dev/null; sleep 5; return; fi
  # The documented server-bench harness -- llama-bench cannot see MTP.
  local R=$(cd $H && timeout 3600 python3 bench/run_bench.py --bench-type server-bench \
      --server-url http://127.0.0.1:$PORT --model qwen27b --bench-configs mtp-dual 2>&1 | grep -aE "_tps:" | tr -s " " | tr "\n" " ")
  echo "$name | ${R:-NO_RESULT}"
  # kill -9 is safe here only because no arm records tune measurements.
  kill -9 $P 2>/dev/null; sleep 6
}
NATIVE=$(find ~/.cache/bigcherry/builds -path "*/a55fa53d6c9c63e01115aa09847f77eb/bin/llama-server" | head -1)
CONTROL=$(find ~/.cache/bigcherry/builds -path "*/df75a6d33c4d2d5342e567ca2a6b01ba/bin/llama-server" | head -1)
REPLAY=$(find ~/.cache/bigcherry/builds -path "*/e59994bc49764809b1b4b957d71e934d/bin/llama-server" | head -1)
cell "llama-native      " "$NATIVE"  no
cell "bc-control(no win)" "$CONTROL" no
cell "bc-REPLAY(19 wins)" "$REPLAY"  yes
echo REPLAY_BENCH_DONE
