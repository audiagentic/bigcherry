#!/bin/bash
# QFP48: what leaves the cards idle between prompt batches (about 32 ms per 512-token batch at 24K, b11552)?
# One uncached Flash-Next prefill under a kernel trace per arm, then kernel-step-census.py's idle breakdown by gap
# size for each: the 1-100 ms rows are the batch boundary.
#   base        the build's defaults
#   nopipe      BIGCHERRY_PREFILL_PIPELINE=0 (1359 off: no prompt batch queued ahead)
#   nocatchup   BIGCHERRY_MTP_DEFERRED_CATCHUP=0 (1348 off)
#   nomtp       no drafter at all
# Usage: queue-boundary-arms.sh <build run> <tag> [depth, default 8192]
set -u
RUN=${1:?build run}; TAG=${2:?tag}; DEPTH=${3:-8192}
here=$(cd "$(dirname "$0")" && pwd)
R=${BC_WORK_ROOT}/runs
( bash "$here/queue-prefill-profile.sh" "$RUN" "$TAG-base" "$DEPTH" )
( export BIGCHERRY_PREFILL_PIPELINE=0; bash "$here/queue-prefill-profile.sh" "$RUN" "$TAG-nopipe" "$DEPTH" )
( export BIGCHERRY_MTP_DEFERRED_CATCHUP=0; bash "$here/queue-prefill-profile.sh" "$RUN" "$TAG-nocatchup" "$DEPTH" )
( export NO_MTP=1; bash "$here/queue-prefill-profile.sh" "$RUN" "$TAG-nomtp" "$DEPTH" )
for arm in base nopipe nocatchup nomtp; do
    echo "######## $arm: $(grep -hE 'prefillprof: prompt' "$R/$TAG-$arm-d$DEPTH.log" | tail -1 | cut -c1-90)"
    grep -hiE "PREFILL_PIPELINE|deferred catch|BIGCHERRY_PATCH_HIT patch=13(48|59)" "$R/$TAG-$arm-d$DEPTH/prefillprof.server.log" 2>/dev/null | sort | uniq -c | head -4 | cut -c1-150
    python3 "$here/kernel-step-census.py" "$R/$TAG-$arm-d$DEPTH/rocprof" 2>&1 | grep -E "^Agent|median step|idle time inside|^    [<>0-9]" | head -9 | cut -c1-230
done
echo ALL_ARMS_DONE
