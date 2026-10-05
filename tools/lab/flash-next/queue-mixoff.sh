#!/bin/bash
# QFP22: 1333 (upstream #29622 mixed token/embd input branch built only for a mixed ubatch) on pin b11402.
# Builds deploy-v6-chunk-mixed-on-demand as <build run>, then compares it with two reference builds in turn
# (Flash-Next 24K MTP ABA + 27B ABBA each): e.g. the previous pin's build (is the old speed recovered?) and the
# new pin's build without the patch (is it faster?).
# Usage: queue-mixoff.sh <build run> <reference build run 1> <reference build run 2> [log to wait for]
set -u
RUN=${1:?build run name for the patched build, e.g. b-mixauto}
REF1=${2:?first reference build run, e.g. b-chunk8}
REF2=${3:?second reference build run, e.g. b-bump3-b11402}
WAIT=${4:-}
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../../.."
[ -n "$WAIT" ] && until grep -q ALL_JOBS_DONE "$WAIT" 2>/dev/null; do sleep 20; done
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
echo "VIS=0,1,2,3 BUILD $RUN bigcherry:stock:linux-multi deploy-v6-chunk-mixed-on-demand gfx1100,gfx1201,gfx1030" > "$jobs"
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "BUILD_QUEUE_EXIT=$?"
rm -f "$jobs"
bash "$HERE/queue-bump-ab.sh" "$REF1" "$RUN" "$RUN-vs-$REF1" | sed 's/^ALL_JOBS_DONE$/PART_DONE/'
bash "$HERE/queue-bump-ab.sh" "$REF2" "$RUN" "$RUN-vs-$REF2"
