#!/bin/bash
# RAD20: where does the burst build's ~0.7% decode cost go? For each arm (two existing builds: main and burst), runs the
# production Flash-Next layout with both helpers under rocprofv3 --kernel-trace, R3_MOE_WORKER_TRACE=1 (per-request
# helper GPU time and engine wait), and samples every helper GPU's shader clock and busy % from sysfs at 20 Hz while a
# fixed decode workload runs. rad20-profile-analyse.py then compares per-kernel durations, per-request times and clocks.
# Devices from the environment: XTX_BDFS (helper GPU PCI addresses). Builds: RAD20_MAIN_BUILD and RAD20_BURST_BUILD.
# Usage: XTX_BDFS=... RAD20_MAIN_BUILD=... RAD20_BURST_BUILD=... rad20-burst-profile.sh [requests=6]
set -u
N=${1:-6}
: "${XTX_BDFS:?set XTX_BDFS}" "${RAD20_MAIN_BUILD:?set RAD20_MAIN_BUILD}" "${RAD20_BURST_BUILD:?set RAD20_BURST_BUILD}"
set --   # the env script reads its own arguments
source tools/env/bigcherry-env.sh >/dev/null
export ROCM_PATH=${ROCM_PATH:-/opt/rocm-7.2.4}
OUT=$BC_WORK_ROOT/runs/rad20-burst-profile-$(date +%m%d%H%M); mkdir -p $OUT
HERE=$(cd "$(dirname "$0")" && pwd)
L=$HERE/launch-flashnext-best.sh
URL=http://127.0.0.1:36080/v1/completions
sampler() {   # $1 csv; one row per GPU per tick: t, bdf, sclk_mhz, busy_pct, mclk_mhz
    echo "t,bdf,sclk_mhz,busy_pct,mclk_mhz" > "$1"
    while :; do
        local t=$(date +%s.%N) b f m
        for b in $XTX_BDFS; do
            f=$(cat /sys/bus/pci/devices/$b/hwmon/hwmon*/freq1_input 2>/dev/null)
            m=$(cat /sys/bus/pci/devices/$b/hwmon/hwmon*/freq2_input 2>/dev/null)
            echo "$t,$b,$(( ${f:-0} / 1000000 )),$(cat /sys/bus/pci/devices/$b/gpu_busy_percent 2>/dev/null),$(( ${m:-0} / 1000000 ))" >> "$1"
        done
        sleep 0.05
    done
}
for arm in main burst; do
    B=$RAD20_MAIN_BUILD; [ $arm = burst ] && B=$RAD20_BURST_BUILD
    D=$OUT/$arm; mkdir -p $D
    wrap=$D/helper-rocprof; cat > $wrap <<EOF
#!/bin/bash
exec rocprofv3 --kernel-trace --output-format csv -d $D/trace\$R3_MOE_WORKER_INDEX -o run -- "\$@"
EOF
    chmod +x $wrap
    HELPER_WRAP=$wrap R3_MOE_WORKER_TRACE=1 bash $L start --build $B > $D/launch.log 2>&1 || { tail $D/launch.log; bash $L stop; exit 1; }
    grep -oE "log [^ ]*\.log" $D/launch.log | cut -d' ' -f2 | sort -u > $D/logs.txt
    curl -s $URL -d '{"prompt":"Warm up.","max_tokens":32,"temperature":0}' > /dev/null
    sleep 3
    sampler $D/clocks.csv & SP=$!
    date +%s.%N > $D/window.txt
    for i in $(seq $N); do
        curl -s $URL -d "{\"prompt\":\"Write a long story about ship number $i.\",\"max_tokens\":512,\"temperature\":0}" \
            | python3 -c "import json,sys; u=json.load(sys.stdin).get('usage',{}); print(u.get('completion_tokens'))" >> $D/tokens.txt
        sleep 2   # idle gap between requests, as between turns
    done
    date +%s.%N >> $D/window.txt
    kill $SP
    bash $L stop > $D/stop.log 2>&1
    for f in $(cat $D/logs.txt); do cp $f $D/; done
done
python3 $HERE/rad20-profile-analyse.py $OUT | tee $OUT/summary.txt
echo "OUT $OUT"; echo BURST_PROFILE_DONE
