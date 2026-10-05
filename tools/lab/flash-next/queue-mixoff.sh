#!/bin/bash
# QFP22: 1333 (upstream #29622 mixed token/embd input branch built only on request) on pin b11402.
# Builds deploy-v6-chunk-mixed-off, then compares it with the previous pin's build (does it recover the old speed?)
# and with the b11402 build without the patch (does it beat it?): Flash-Next 24K MTP ABA + 27B ABBA each.
# Usage: queue-mixoff.sh <previous-pin build run> <new-pin build run> [log to wait for]
set -u
OLD=${1:?previous-pin build run, e.g. b-chunk8}
NEW=${2:?new-pin build run without 1333, e.g. b-bump3-b11402}
WAIT=${3:-}
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../../.."
R=/mnt/data/bigcherry-work/runs
[ -n "$WAIT" ] && until grep -q ALL_JOBS_DONE "$WAIT" 2>/dev/null; do sleep 20; done
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
echo "VIS=0,1,2,3 BUILD b-mixoff bigcherry:stock:linux-multi deploy-v6-chunk-mixed-off gfx1100,gfx1201,gfx1030" > "$jobs"
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "BUILD_QUEUE_EXIT=$?"
rm -f "$jobs"
bash "$HERE/queue-bump-ab.sh" "$OLD" b-mixoff mixoff-vs-old | sed 's/^ALL_JOBS_DONE$/PART_DONE/'
bash "$HERE/queue-bump-ab.sh" "$NEW" b-mixoff mixoff-vs-new
