#!/bin/bash
# 27B dual-XTX production ABBA (prod27b-ab.sh, 10K + 32K) for a list of build-run pairs, each repeated N times.
# Used to place a small residual against existing bisect binaries without building anything.
# Usage: queue-27b-pairs.sh <tag> <repeats> <A:B>... [wait=<log with ALL_JOBS_DONE>]
set -u
TAG=${1:?tag}
REP=${2:?repeats}
shift 2
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=/mnt/data/bigcherry-work/runs
for a in "$@"; do case "$a" in wait=*) until grep -q "^ALL_JOBS_DONE" "${a#wait=}" 2>/dev/null; do sleep 20; done ;; esac; done
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
for r in $(seq 1 "$REP"); do
    for a in "$@"; do
        case "$a" in wait=*) continue ;; esac
        A=${a%%:*}; B=${a##*:}
        echo "VIS=0,1 SCRIPT $TAG-$A-vs-$B-r$r tools/lab/flash-next/prod27b-ab.sh @$A @$B $R/$TAG-$A-vs-$B-r$r" >> "$jobs"
    done
done
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
for a in "$@"; do
    case "$a" in wait=*) continue ;; esac
    A=${a%%:*}; B=${a##*:}
    echo "== A = $A, B = $B"
    for r in $(seq 1 "$REP"); do grep -E "^d[0-9]|SERVER_FAILED" "$R/$TAG-$A-vs-$B-r$r.log"; done
done
echo ALL_JOBS_DONE
