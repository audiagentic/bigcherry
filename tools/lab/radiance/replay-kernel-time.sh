#!/bin/bash
# Per-kernel GPU time of the remote-MoE worker on real captured requests (RAD14): ONE worker process replays the captures under
# rocprofv3's kernel trace and the script prints, per kernel name, the call count and mean duration. Run it per configuration
# (SMALL_ROWS, GEMV settings, device) and compare. No server, no engine: kernel time only, so it separates kernel cost from
# orchestration cost when the same captures are replayed on another card (DEV).
# Usage: BUILD=<lane build dir> CAPDIR=<captures> MODEL=<.rad> [DEV=0] [N=12] [ENVV="R3_MOE_WORKER_SMALL_ROWS=16 ..."] tools/lab/radiance/replay-kernel-time.sh
set -u
B=${BUILD:?set BUILD to the lane build directory}
CAP=${CAPDIR:?set CAPDIR to a directory of captures}
M=${MODEL:?set MODEL to the .rad container the captures came from}
DEV=${DEV:-0}
OUT=${OUT:-${WORK:-/mnt/data/bigcherry-work}/replay-kt}
FILES=$(ls ${CAP}/*.bin | head -${N:-12} | tr '\n' ' ')
export ROCM_PATH=${ROCM_PATH:-/opt/rocm-7.2.4} PATH=${ROCM_PATH:-/opt/rocm-7.2.4}/bin:$PATH
rm -rf "$OUT"
env HIP_VISIBLE_DEVICES=$DEV R3_MOE_WORKER_MIB=${MIB:-22528} R3_MOE_WORKER_MAX_ROWS=4096 R3_MOE_WORKER_SMALL_ROWS=${SMALL_ROWS:-8} R3_MOE_WORKER_ENGINE_RANKS=${ENGINE_RANKS:-1} R3_MOE_WORKER_MOE_DOWN_TILE=1 R3_MOE_SWEEP_FAST=1 RADIANCE_HOME=$B/radiance_home ${ENVV:-} \
  timeout 3600 rocprofv3 --kernel-trace --output-format csv -d "$OUT" -o run -- \
  "$B/radiance_home/helpers/r3_moe_worker" --replay-sweep "$M" "" $FILES > "$OUT.log" 2>&1
echo "rc=$? ($(grep -c '^replay_hash' "$OUT.log") replays)"
python3 - "$(find "$OUT" -name '*kernel_trace.csv' | head -1)" <<'PY'
import csv, sys, collections
agg = collections.defaultdict(lambda: [0, 0])
for r in csv.DictReader(open(sys.argv[1])):
    n = r['Kernel_Name']
    if not any(s in n for s in ('lane_', 'moe_gemm', 'moe_blockmap', 'quant_fp8')):
        continue
    a = agg[n.split('(')[0][:70]]
    a[0] += 1
    a[1] += int(r['End_Timestamp']) - int(r['Start_Timestamp'])
for n, (c, t) in sorted(agg.items(), key=lambda x: -x[1][1]):
    print('%-72s calls %6d  mean %8.1f us  total %9.1f us' % (n, c, t / c / 1e3, t / 1e3))
PY
