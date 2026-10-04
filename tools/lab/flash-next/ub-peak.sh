#!/bin/bash
# QFP17: ub 512 then 1024 load with the live-at-peak allocator trace (1331) and the v6 profile via one feature-set
# flag (0910 end-to-end check). Usage: ub-peak.sh <bin> <root>
set -u
unset GGML_HIP_Q8_1_CACHE_MODE BIGCHERRY_ROLLBACK_NO_CONT BIGCHERRY_RMS_Q81 BIGCHERRY_ACT_Q81 BIGCHERRY_HC_Q81 \
      BIGCHERRY_SCALE_ACT_FUSE BIGCHERRY_SCHED_ASYNC_INPUTS BIGCHERRY_QSA_HOST_REMAP
export BIGCHERRY_FEATURES=flashnext BIGCHERRY_ALLOC_PEAK=24 BIGCHERRY_ALLOC_TOP=8
mkdir -p "$2"
BIGCHERRY_FEATURES=help "$1" > "$2/features-help.txt" 2>&1
echo "help rc=$? lines=$(wc -l < "$2/features-help.txt")"
exec bash "$(cd "$(dirname "$0")" && pwd)/ub-sweep.sh" "$1" "$2" 512 1024
