#!/bin/bash
# RAD20 stage A: cross-process BARRIER_VALUE correctness/latency (R9700 producer -> XTX consumer), then XTX idle power and
# GFX activity with 0/1/2/4/8 queues held at an unsatisfied BARRIER_VALUE (amd-smi every second), baseline again at the end.
# Usage: barrier-value-stage-a.sh <build dir with barrier-value-bench>   (XTX = ROCR device 0 / amd-smi GPU 0, R9700 = 2)
set -u
B=${1:?bench binary directory}
OUT=${BC_WORK_ROOT}/runs/rad20-stage-a-$(date +%m%d%H%M); mkdir -p $OUT
cd $B
echo "== correctness / latency" | tee -a $OUT/summary.txt
[ "${KIND:-and}" = value ] && for cfg in "2000 150" "1000 1000"; do
    set -- $cfg
    ROCR_VISIBLE_DEVICES=0 timeout 120 ./barrier-value-bench consumer-latency $1 4 > $OUT/cons.txt 2>&1 &
    sleep 1
    ROCR_VISIBLE_DEVICES=2 timeout 120 ./barrier-value-bench producer $1 $2 > $OUT/prod.txt 2>&1
    wait
    echo "-- $1 notifications every $2 us" | tee -a $OUT/summary.txt
    cat $OUT/cons.txt $OUT/prod.txt | grep -v '^$' | tee -a $OUT/summary.txt
done
sample() {   # $1 label, $2 seconds: amd-smi GPU 0 power and GFX activity once a second
    for _ in $(seq $2); do
        amd-smi metric -g 0 -p -u -c 2>/dev/null | awk -v l=$1 '/SOCKET_POWER/{p=$2} /GFX_ACTIVITY/{a=$2} /GFX_0:/{g=1} g&&/CLK:/{c=$2;g=0} END{print l, p, a, c}'
        sleep 1
    done
}
echo "== idle: label, board W, GFX activity %, gfx clock" | tee -a $OUT/summary.txt
sample base0 20 > $OUT/idle.txt
for qn in 1 2 4 8; do
    ROCR_VISIBLE_DEVICES=0 ./barrier-value-bench idle $qn 30 ${KIND:-and} >> $OUT/idle-runs.txt 2>&1 &
    sleep 5
    sample q$qn 20 >> $OUT/idle.txt
    wait
done
sample base1 20 >> $OUT/idle.txt
cat $OUT/idle-runs.txt | tee -a $OUT/summary.txt
awk '{n[$1]++; w[$1]+=$2; a[$1]+=$3} END{for (k in n) printf "%s: %.1f W, GFX %.1f %% (%d samples)\n", k, w[k]/n[k], a[k]/n[k], n[k]}' $OUT/idle.txt | sort | tee -a $OUT/summary.txt
echo "OUT $OUT"; echo STAGE_A_DONE
