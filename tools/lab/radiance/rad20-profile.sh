#!/bin/bash
# RAD20 decode diagnosis: why do the helper's small-lane kernels run ~2x slower with blocking lanes? For each build
# (polling baseline, RAD20 blocking): start the best layout with helper 0 under rocprofv3 --kernel-trace, sample XTX 0's
# clocks, send one 512-token decode request, stop. Compares per-kernel device time and clocks between the two.
# Usage: rad20-profile.sh <polling build plan id> <blocking build plan id>   (directories under $BC_WORK_ROOT/builds)
set -u
POLL=$1 BLOCK=$2 LABEL=${3:-auto}
set --
source tools/env/bigcherry-env.sh
W=${BC_WORK_ROOT}; OUT=$W/runs/rad20-profile-$LABEL-$(date +%m%d%H%M); mkdir -p $OUT
export ROCM_PATH=/opt/rocm-7.2.4 PATH=/opt/rocm-7.2.4/bin:$PATH
L=tools/lab/radiance/launch-flashnext-best.sh
PROMPT='Write a long, detailed story about a lighthouse keeper who discovers a hidden room. Keep going.'
for arm in polling:$POLL blocking:$BLOCK; do
    name=${arm%%:*}; build=$(ls -d $W/builds/*/${arm#*:} | head -1)
    HELPER0_WRAP="rocprofv3 --kernel-trace --stats --output-format csv -d $OUT/$name-prof --" \
        bash $L start --build $build > $OUT/$name-launch.log 2>&1 || { tail $OUT/$name-launch.log; bash $L stop; continue; }
    # one warm-up, then the measured decode with clocks sampled (1 s) on XTX 0
    curl -s http://127.0.0.1:36080/v1/completions -d "{\"prompt\":\"$PROMPT\",\"max_tokens\":64}" > /dev/null
    # XTX 0's current sclk/mclk levels from sysfs every 10 ms (the first card whose pp_dpm_mclk tops out at 1249 MHz)
    CARD=$(grep -l '1249Mhz' /sys/class/drm/card*/device/pp_dpm_mclk | head -1 | xargs dirname)
    ( while :; do echo "mclk $(grep '\*' $CARD/pp_dpm_mclk) sclk $(grep '\*' $CARD/pp_dpm_sclk)"; sleep 0.01; done ) > $OUT/$name-clocks.txt &
    CL=$!
    t0=$(date +%s.%N)
    curl -s http://127.0.0.1:36080/v1/completions -d "{\"prompt\":\"$PROMPT\",\"max_tokens\":512,\"temperature\":0}" > $OUT/$name-reply.json
    t1=$(date +%s.%N)
    kill $CL 2>/dev/null
    python3 -c "import json,sys;r=json.load(open('$OUT/$name-reply.json'));n=r['usage']['completion_tokens'];print('$name', n, 'tokens', round(n/($t1-$t0),1), 'tok/s (incl prefill)')" | tee -a $OUT/summary.txt
    bash $L stop > /dev/null 2>&1
    cp $W/flashnext/helper0-*.log $OUT/ 2>/dev/null; rm -f $W/flashnext/helper*-*.log $W/flashnext/engine-*.log
done
for name in polling blocking; do
    s=$(find $OUT/$name-prof -name '*kernel_stats.csv' | head -1)
    echo "== $name: top kernels (calls, total ms, avg us)" | tee -a $OUT/summary.txt
    [ -n "$s" ] && python3 - "$s" <<'PY' | tee -a $OUT/summary.txt
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
rows.sort(key=lambda r: -float(r["TotalDurationNs"]))
for r in rows[:10]:
    print(f'{r["Calls"]:>8} {float(r["TotalDurationNs"])/1e6:9.1f} ms {float(r["AverageNs"])/1e3:8.1f} us  {r["Name"][:90]}')
PY
    echo "-- XTX 0 clock levels during the decode (10 ms samples: count, mclk, sclk)" | tee -a $OUT/summary.txt
    sed -E 's/ +/ /g' $OUT/$name-clocks.txt | sort | uniq -c | sort -rn | head -8 | tee -a $OUT/summary.txt
done
echo "OUT $OUT"; echo RAD20_PROFILE_DONE
