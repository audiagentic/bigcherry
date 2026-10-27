#!/bin/bash
# RAD20: where does the burst build's ~0.7% decode cost go? For each arm (two existing builds: main and burst), runs the
# production Flash-Next layout with both helpers under rocprofv3 --kernel-trace, R3_MOE_WORKER_TRACE=1 (per-request
# helper GPU time and engine wait), and samples every helper GPU's shader clock and busy % from sysfs at 20 Hz while a
# fixed decode workload runs. rad20-profile-analyse.py then compares per-kernel durations, per-request times and clocks.
# Devices from the environment: XTX_BDFS (helper GPU PCI addresses). Builds: RAD20_MAIN_BUILD and RAD20_BURST_BUILD.
# Optional helper-only environment per arm (e.g. GPU_MAX_HW_QUEUES=13): RAD20_HELPER_ENV_MAIN / RAD20_HELPER_ENV_BURST;
# environment for the engine and the helpers both (e.g. R3_MOE_WORKER_COMBINE=1): RAD20_ARM_ENV_MAIN / _BURST. The two
# "arms" may then be the same build. Each arm's build and environment is recorded in arms.txt.
# RAD20_WORKLOAD: "decode" (default; N stories of 512 tokens with 2 s gaps) or "prefill-decode" (N prompts of about 1,000
# tokens, 128 tokens each - the bench's tg128 shape - through rad20-decode-client.py, which records the server's prefill
# and decode timings, MTP draft acceptance and a hash of each generated text; prompt text from RAD20_PROMPT_FILE).
# Either way the helpers' per-thread CPU use is sampled once a second (top -H) while the workload runs.
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
    HENV=${RAD20_HELPER_ENV_MAIN:-}; [ $arm = burst ] && HENV=${RAD20_HELPER_ENV_BURST:-}
    AENV=${RAD20_ARM_ENV_MAIN:-}; [ $arm = burst ] && AENV=${RAD20_ARM_ENV_BURST:-}
    echo "$arm build=$B helper_env=$HENV arm_env=$AENV workload=${RAD20_WORKLOAD:-decode}" >> $OUT/arms.txt
    wrap=$D/helper-rocprof; cat > $wrap <<EOF
#!/bin/bash
${HENV:+export $HENV}
exec rocprofv3 --kernel-trace --output-format csv -d $D/trace\$R3_MOE_WORKER_INDEX -o run -- "\$@"
EOF
    chmod +x $wrap
    env $AENV HELPER_WRAP=$wrap R3_MOE_WORKER_TRACE=1 bash $L start --build $B > $D/launch.log 2>&1 || { tail $D/launch.log; bash $L stop; exit 1; }
    grep -oE "log [^ ]*\.log" $D/launch.log | cut -d' ' -f2 | sort -u > $D/logs.txt
    curl -s $URL -d '{"prompt":"Warm up.","max_tokens":32,"temperature":0}' > /dev/null
    sleep 3
    sampler $D/clocks.csv & SP=$!
    HPIDS=$(pgrep -d, -f "radiance_home/helpers/r3_moe_worker")
    top -H -b -d 1 -p "$HPIDS" > $D/helper-threads.top 2>/dev/null & TP=$!
    date +%s.%N > $D/window.txt
    if [ "${RAD20_WORKLOAD:-decode}" = prefill-decode ]; then
        python3 $HERE/rad20-decode-client.py $URL "${RAD20_PROMPT_FILE:?set RAD20_PROMPT_FILE}" $N 4000 128 $D/requests.jsonl
        python3 -c "import json; print('\n'.join(str(json.loads(l)['predicted_n']) for l in open('$D/requests.jsonl')))" > $D/tokens.txt
    else
        for i in $(seq $N); do
            curl -s $URL -d "{\"prompt\":\"Write a long story about ship number $i.\",\"max_tokens\":512,\"temperature\":0}" \
                | python3 -c "import json,sys; u=json.load(sys.stdin).get('usage',{}); print(u.get('completion_tokens'))" >> $D/tokens.txt
            sleep 2   # idle gap between requests, as between turns
        done
    fi
    date +%s.%N >> $D/window.txt
    kill $SP $TP
    bash $L stop > $D/stop.log 2>&1
    for f in $(cat $D/logs.txt); do cp $f $D/; done
done
python3 $HERE/rad20-profile-analyse.py $OUT | tee $OUT/summary.txt
echo "OUT $OUT"; echo BURST_PROFILE_DONE
