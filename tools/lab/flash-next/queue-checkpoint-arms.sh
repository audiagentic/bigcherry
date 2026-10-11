#!/bin/bash
# QFP48: what do the server's context checkpoints cost a prefill?
# The sync tracer shows the server taking a checkpoint several times during a prompt (llama_state_seq_get_data_ext
# from common_prompt_checkpoint::update): one wait for the queued batches, then a few hundred small synchronous
# tensor reads, with nothing queued on the cards meanwhile. One uncached Flash-Next prefill under a kernel trace per
# arm; the 1-100 ms rows of the idle breakdown and the prompt speed are the comparison.
#   base      the server's defaults (32 checkpoints, at least 8192 tokens apart)
#   nockpt    --ctx-checkpoints 0
#   every4k   --checkpoint-min-step 4096 (twice as many)
# Usage: queue-checkpoint-arms.sh <build run> <tag> [depth, default 24576]
set -u
RUN=${1:?build run}; TAG=${2:?tag}; DEPTH=${3:-24576}
here=$(cd "$(dirname "$0")" && pwd)
R=/mnt/data/bigcherry-work/runs
( bash "$here/queue-prefill-profile.sh" "$RUN" "$TAG-base" "$DEPTH" )
( export EXTRA_ARGS="--ctx-checkpoints 0"; bash "$here/queue-prefill-profile.sh" "$RUN" "$TAG-nockpt" "$DEPTH" )
( export EXTRA_ARGS="--checkpoint-min-step 4096"; bash "$here/queue-prefill-profile.sh" "$RUN" "$TAG-every4k" "$DEPTH" )
for arm in base nockpt every4k; do
    echo "######## $arm: $(grep -hE 'prefillprof: prompt' "$R/$TAG-$arm-d$DEPTH.log" | tail -1 | cut -c1-90)"
    echo "  checkpoints created: $(grep -ciE 'created context checkpoint|checkpoint [0-9]+ of' "$R/$TAG-$arm-d$DEPTH/prefillprof.server.log" 2>/dev/null)"
    python3 "$here/kernel-step-census.py" "$R/$TAG-$arm-d$DEPTH/rocprof" 2>&1 | grep -E "^Agent|median step|idle time inside|^    [<>0-9]" | head -9 | cut -c1-230
done
echo ALL_ARMS_DONE
