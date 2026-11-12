#!/bin/bash
# Per-op profile of a Radiance server over fixed workloads, then where the time goes per rank (tp-profile-split.py):
# trunk compute, drafter, collectives, and the top ops. Each workload runs on a fresh start with --profile-ops, which
# restores synchronisations, so its times are for comparing configurations profiled the same way, not with unprofiled
# throughput. Run it beside an experiment's unprofiled arms, under the gpu lease.
#
# Usage: tp-profile.sh BUILD_DIR MODEL OUT_DIR NAME [WORKLOADS] -- ENGINE_ARGS...
#   WORKLOADS  comma list of dec1, dec8 (512-token prompts, 128 new), pre1 (one 4096-token prompt),
#              pre4 (four 4096-token prompts at once); default dec8,pre1,pre4
#   PORT       env, default 18405
set -u
[ $# -ge 5 ] || { sed -n 2,12p "$0"; exit 2; }
B=$1 MODEL=$2 OUT=$3 NAME=$4; shift 4
WL=dec8,pre1,pre4
[ "$1" != "--" ] && { WL=$1; shift; }
[ "$1" = "--" ] && shift
PORT=${PORT:-18405}
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd)
mkdir -p "$OUT"
bench() { (cd "$REPO" && PYTHONPATH=tools timeout 900 python3 -m bigcherry stream-bench --url http://127.0.0.1:$PORT --model qwen35 "$@"); }
for wl in ${WL//,/ }; do
  tag=$NAME-$wl
  echo "== $(date -u +%H:%M:%SZ) $tag start" | tee -a "$OUT/steps.log"
  RADIANCE_HOME=$B/radiance_home setsid "$B/bin/radiance" --model "$MODEL" --host 127.0.0.1 --port $PORT --profile-ops "$@" \
    > "$OUT/$tag.serve.log" 2>&1 &
  sp=$! ok=0
  for i in $(seq 1 120); do sleep 5; curl -sf http://127.0.0.1:$PORT/health >/dev/null && { ok=1; break; }; kill -0 $sp 2>/dev/null || break; done
  if [ $ok = 1 ]; then
    case $wl in
      dec1) bench --streams 1 --prompt-tokens 512 --max-tokens 128 --reps 1 --temperature 0 --seed 7 --output "$OUT/$tag.json" ;;
      dec8) bench --streams 8 --prompt-tokens 512 --max-tokens 128 --reps 1 --temperature 0 --seed 7 --output "$OUT/$tag.json" ;;
      pre1) bench --streams 1 --prompt-tokens 4096 --max-tokens 4 --reps 2 --temperature 0 --seed 9 --output "$OUT/$tag.json" ;;
      pre4) bench --streams 4 --prompt-tokens 4096 --max-tokens 4 --reps 1 --temperature 0 --seed 9 --output "$OUT/$tag.json" ;;
      *) echo "unknown workload $wl" >&2 ;;
    esac > "$OUT/$tag.bench.log" 2>&1
  else echo "== $tag FAILED to start" | tee -a "$OUT/steps.log"; fi
  # The profile tables are printed at shutdown: stop with INT and give the dump time.
  kill -INT $sp 2>/dev/null
  for i in $(seq 1 60); do sleep 1; kill -0 $sp 2>/dev/null || break; done
  kill -0 $sp 2>/dev/null && { echo "== $tag did not exit; killed (no profile)" | tee -a "$OUT/steps.log"; kill -KILL $sp; }
  sleep 3
  python3 "$HERE/tp-profile-split.py" "$OUT/$tag.serve.log" | tee -a "$OUT/split.txt"
done
