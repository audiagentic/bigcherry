#!/bin/bash
# RR07/RR08: where does the gfx11 MXFP4 decode kernel (r3_mxfp4_dec11.h) spend its time? One profiled serve run per
# R3_D11_MODE value: bit 1 switches off the matrix instruction, bit 2 the 4-bit unpack, bit 4 the weight loads. The
# replies are WRONG in every mode but 0; only the per-op times are read. Prints, per mode, the mean time of each
# MXFP4 GEMM shape from the engine's --profile-ops table.
# Runs the serve jobs through the queue itself (one card), so start it directly, not as a queue job:
#   bash tools/lab/radiance/libr3-dec11-ablation.sh [mode...]        (default: 0 1 2 3 4 5 6 7)
# env: R (runs root), VIS (0), BUILD (any build run id for the queue line, b-main2), N (64 tokens)
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=${R:-/mnt/data/bigcherry-work/runs}
modes=("$@"); [ ${#modes[@]} -eq 0 ] && modes=(0 1 2 3 4 5 6 7)
jobs=$(mktemp)
for mode in "${modes[@]}"; do
    run=r3-abl-$mode
    rm -rf "$R/$run" "$R/$run.log"
    echo "VIS=${VIS:-0} SCRIPT $run tools/lab/radiance/libr3-serve-smoke.sh @${BUILD:-b-main2} $R/$run" > "$jobs"
    env N="${N:-64}" R3_DEC11=1 R3_D11_MODE="$mode" EXTRA=--profile-ops RADIANCE_PROFILE_EVERY=64 \
        bash tools/lab/plan-qualification/queue.sh "$jobs" > /dev/null 2>&1
    log=$R/$run/server.log
    if ! grep -q "per-op timing, rank 0" "$log" 2> /dev/null; then
        echo "mode $mode: NO PROFILE ($(grep -E "SERVER_EXITED|LOAD_TIMEOUT|exit" "$R/$run.log" 2> /dev/null | tail -1 | cut -c1-80))"
        continue
    fi
    start=$(grep -n "per-op timing, rank 0" "$log" | tail -1 | cut -d: -f1)
    echo "mode $mode ($(grep -oE "[0-9.]+ tok/s" "$R/$run.log" | head -1)): $(tail -n +"$start" "$log" | awk '
        $4 == "dev" && $6 == "gemm_nt_q" { k = $7 " " $8; t[k] += $2; c[k] += $1 }
        END { n = asorti(t, order); for (i = 1; i <= n; i++) { k = order[i]; printf "%s %.1f us; ", k, 1000 * t[k] / c[k] } }')"
done
rm -f "$jobs"
echo DEC11_ABLATION_DONE
