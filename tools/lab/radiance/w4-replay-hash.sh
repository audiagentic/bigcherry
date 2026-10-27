#!/bin/bash
# Cross-build identity check for the remote-MoE worker on W4 (Flash-Next): replay real captured requests in ONE worker process and
# print a hash of the helper's result bytes for each (RCS05). Run it on a baseline build and on a candidate build with the same
# captures and settings; every replay_hash line must match. (rel_l2 only bounds the error; this proves bit identity.)
# Usage: BUILD=<lane build dir> [CAPDIR=<dir>] [N=4] tools/lab/radiance/w4-replay-hash.sh   (set MODEL for another container)
set -u
B=${BUILD:?set BUILD to the lane build directory}
W=${WORK:-${BC_WORK_ROOT}}
M=${MODEL:-${BC_MODEL_ROOT}/radiance/StillDeadcode/qwen3.8-next-flash-fp8-iq4r-moe/qwen3.8-next-flash-fp8-iq4r-moe.rad}
FILES=$(ls -S ${CAPDIR:-$W/captures}/*.bin | head -${N:-4} | tr '\n' ' ')
source "$(dirname "${BASH_SOURCE[0]}")/lab-env.sh"
env HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 R3_MOE_WORKER_MIB=22528 R3_MOE_WORKER_MAX_ROWS=4096 R3_MOE_SWEEP_FAST=1 \
  RADIANCE_HOME=$B/radiance_home \
  timeout 3600 "$B/radiance_home/helpers/r3_moe_worker" --replay-sweep "$M" "" $FILES 2>&1 | grep -E "^SWEEP|^replay_hash"
