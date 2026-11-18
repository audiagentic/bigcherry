#!/bin/bash
# VTP44 layout smoke: one layout x KV types. Real launcher, layout as its server-args define it, only --kv-dtype changes.
# Usage: smoke_layout.sh LAYOUT [kv ...]   (default: fp8 bf16_fp8 bf16_i8 bf16; fp8 is the layouts' own type, the baseline)
# Per config: start (the launcher measures a new config first), canary, a short client (8k TTFT, 1-stream decode, 2 needle probes), stop.
# A VRAM admission refusal (gpu_room) is retried once with --max-num-seqs 4 (a smaller sessions set: the launcher default is 8 x 262144 tokens
# of worst-case KV, ~35 GB at bf16 on the 32 GB R9700); that result is labelled <kv>@seqs4, and fp8 is then also run at 4 sessions so the
# packed types compare like for like. If the first type does not start at all the layout is broken and the rest are skipped. Run under the gpu lease.
set -u
L=$1; shift
KVS=("$@"); [ ${#KVS[@]} = 0 ] && KVS=(fp8 bf16_fp8 bf16_i8 bf16)
F=<dev-root>/tmp/bigcherry/vtp44-fn
O=$F/smoke/$L; mkdir -p "$O"
B=<work-root>/builds/b893f5876bb7781648f3dc09390e201c/e5a91ea014057f2e28efee1e6db10b22
MAIN=<dev-root>/projects/bigcherry/workspaces/main
HEAT=$MAIN/tools/lab/radiance/heat/perf51.rank0.hot-first
PORT=18410
export BIGCHERRY_ENVIRONMENT=$MAIN/config/environment.local.toml BIGCHERRY_LAYOUTS=$MAIN/engines/radiance/layouts
cd "$MAIN" || exit 2
set --; source tools/env/bigcherry-env.sh
echo "start_utc $(date -u +%FT%TZ) layout $L build $B" >> "$O/receipt.txt"
one() { # kv [seqs]
  local kv=$1 seqs=${2:-} tag=$1 t0=$(date +%s); local extra=()
  [ -n "$seqs" ] && { tag="$1@seqs$seqs"; extra=(--max-num-seqs "$seqs"); }
  timeout 2400 bash tools/lab/radiance/launch-flashnext-best.sh start --build "$B" --layout "$L" --heat "$HEAT" --port $PORT --ngram-placement disk --kv-dtype $kv "${extra[@]}" > "$O/$tag.launch.log" 2>&1
  local rc=$?
  echo "$(date -u +%T) $tag start exit $rc $(( $(date +%s) - t0 ))s" | tee -a "$O/receipt.txt"
  if [ $rc = 0 ] && curl -sf http://127.0.0.1:$PORT/health >/dev/null; then
    timeout 900 python3 $F/fnsmoke.py http://127.0.0.1:$PORT "$O/$tag.json" > "$O/$tag.client.log" 2>&1
    echo "$(date -u +%T) $tag client exit $?" | tee -a "$O/receipt.txt"
  fi
  bash tools/lab/radiance/launch-flashnext-best.sh stop > "$O/$tag.stop.log" 2>&1
  echo "$(date -u +%T) $tag stop exit $?" | tee -a "$O/receipt.txt"
  mkdir -p "$O/$tag.runlogs"; cp -a "$BC_WORK_ROOT"/flashnext/*.log "$O/$tag.runlogs/" 2>/dev/null
  if [ $rc != 0 ]; then
    echo "  $tag refused/failed: $(grep -o 'reason_code":"[a-z_]*' "$O/$tag.launch.log" | head -1) $(grep -o 'shortfall_bytes":[0-9]*' "$O/$tag.launch.log" | head -1)" | tee -a "$O/receipt.txt"
  fi
  return $rc
}
RETRIED=0; SKIPPED=0
for kv in "${KVS[@]}"; do
  one $kv; rc=$?
  if [ $rc != 0 ] && grep -q 'reason_code":"gpu_room' "$O/$kv.launch.log"; then
    echo "  $kv does not fit at the layout's sessions: retry with 4 sessions" | tee -a "$O/receipt.txt"
    RETRIED=1; one $kv 4; rc=$?
  fi
  if [ $rc != 0 ] && [ $kv = "${KVS[0]}" ]; then echo "SKIP: the first type ($kv) did not start; layout $L is broken, not the cache" | tee -a "$O/receipt.txt"; SKIPPED=1; break; fi
done
if [ $RETRIED = 1 ] && [ $SKIPPED = 0 ] && printf '%s\n' "${KVS[@]}" | grep -qx fp8; then one fp8 4; fi
echo "end_utc $(date -u +%FT%TZ)" >> "$O/receipt.txt"
