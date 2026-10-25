#!/bin/bash
# RAD20 decode diagnosis: why do the helper's small-lane kernels run ~2x slower with blocking lanes? For each build
# (polling baseline, RAD20 blocking): start the best layout with helper 0 under rocprofv3 --kernel-trace, sample XTX 0's
# clocks, send one 512-token decode request, stop. Compares per-kernel device time and clocks between the two.
# Usage: rad20-profile.sh <polling build dir> <blocking build dir>
set -u
W=${BC_WORK_ROOT}; OUT=$W/runs/rad20-profile-$(date +%m%d%H%M); mkdir -p $OUT
export ROCM_PATH=/opt/rocm-7.2.4 PATH=/opt/rocm-7.2.4/bin:$PATH
L=tools/lab/radiance/launch-flashnext-best.sh
PROMPT='Write a long, detailed story about a lighthouse keeper who discovers a hidden room. Keep going.'
for arm in polling:$1 blocking:$2; do
    name=${arm%%:*}; build=${arm#*:}
    HELPER0_WRAP="rocprofv3 --kernel-trace --stats --output-format csv -d $OUT/$name-prof --" \
        bash $L start --build $build > $OUT/$name-launch.log 2>&1 || { tail $OUT/$name-launch.log; bash $L stop; continue; }
    # one warm-up, then the measured decode with clocks sampled (1 s) on XTX 0
    curl -s http://127.0.0.1:36080/v1/completions -d "{\"prompt\":\"$PROMPT\",\"max_tokens\":64}" > /dev/null
    ( for _ in $(seq 40); do date +%s.%N; rocm-smi -d 0 --showclocks --showuse --showpower 2>/dev/null | grep -E 'sclk|mclk|fclk|GPU use|Power'; sleep 1; done ) > $OUT/$name-clocks.txt &
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
    echo "-- clocks (median of samples)" | tee -a $OUT/summary.txt
    grep -oE '(sclk|mclk|fclk) clock level: [0-9]+: \([0-9]+Mhz\)' $OUT/$name-clocks.txt | sed -E 's/ clock level: [0-9]+: \(/ /; s/Mhz\)//' | sort | awk '{a[$1]=a[$1]" "$2} END{for(k in a){n=split(a[k],v," "); asort(v); print k, "median", v[int((n+1)/2)], "MHz, samples", n}}' | tee -a $OUT/summary.txt
done
echo "OUT $OUT"; echo RAD20_PROFILE_DONE
