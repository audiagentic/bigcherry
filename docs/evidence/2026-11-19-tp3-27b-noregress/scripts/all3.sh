#!/bin/bash
# replay proof: bc-serve from the s8-inventory-identity branch, two consecutive starts of one config in ONE run dir,
# 20 s apart (the gap that failed with the old digest); the second must replay the first's stored manifest
D=<dev-root>/tmp/bigcherry/tp3-check
export RUN_DIR=<work-root>/tp3check-replay QUICK=1
rm -rf "$RUN_DIR"
for n in 1 2; do
  bash $D/one.sh $D/fixbuild $D/r$n qwen27b-tp3-int8-556-long-bf16
  echo "replay start $n exit=$? at $(date -Is)" >> $D/replay.log
  sleep 20
done
