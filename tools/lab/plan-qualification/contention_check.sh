#!/bin/bash
# Does a job on one GPU perturb a measurement on another? A/A check for
# running campaign lanes in parallel. Measures the SAME llama-bench binary on
# the target device alone and while a neighbour load runs on other devices,
# alternating solo/loaded blocks so drift cannot masquerade as contention.
# Usage: contention_check.sh <llama-bench> <model> <hip-prefix> <target-dev> <load-devs,comma> <load-model> [rounds]
# Run only in an exclusive window (queue stopped between jobs).
set -u
bench=$1; model=$2; hip=$3; target=$4; loads=$5; load_model=$6; rounds=${7:-6}
root=$(cd "$(dirname "$0")/../../.." && pwd)
work=$("$root/tools/lab/plan-qualification/work-root.sh" "$root")
out=$work/runs/contention-check-$(date +%Y%m%dT%H%M%S)
mkdir -p "$out"
export ROCM_PATH=$hip HIP_PATH=$hip
unset ROCR_VISIBLE_DEVICES
measure() {  # $1 = label
    HIP_VISIBLE_DEVICES=$target "$bench" -m "$model" -p 512 -n 128 -ngl 99 -r 3 2>/dev/null \
        | grep -E "pp512|tg128" | sed "s/^/$1 /" >> "$out/table.txt"
}
start_load() {
    pids=()
    for d in ${loads//,/ }; do
        ( while :; do HIP_VISIBLE_DEVICES=$d "$bench" -m "$load_model" -p 512 -n 128 -ngl 99 -r 5 > /dev/null 2>&1; done ) &
        pids+=($!)
    done
    sleep 20  # let the neighbours reach steady state
}
stop_load() {
    for p in "${pids[@]}"; do pkill -P "$p" 2>/dev/null; kill "$p" 2>/dev/null; done
    wait 2>/dev/null
    sleep 10
}
for r in $(seq 1 "$rounds"); do
    if [ $((r % 2)) -eq 1 ]; then measure solo; start_load; measure loaded; stop_load
    else start_load; measure loaded; stop_load; measure solo; fi
done
python3 - "$out/table.txt" <<'PY'
import re, statistics, sys
vals = {}
for line in open(sys.argv[1]):
    label = line.split()[0]
    test = "pp512" if "pp512" in line else "tg128"
    m = re.search(r"\|\s*([0-9.]+)\s*±", line)
    if m:
        vals.setdefault((test, label), []).append(float(m.group(1)))
for test in ("pp512", "tg128"):
    s, l = vals.get((test, "solo"), []), vals.get((test, "loaded"), [])
    if s and l:
        ms, ml = statistics.fmean(s), statistics.fmean(l)
        print(f"{test}: solo {ms:.2f} (sd {statistics.pstdev(s):.2f}, n={len(s)})  "
              f"loaded {ml:.2f} (sd {statistics.pstdev(l):.2f}, n={len(l)})  delta {100*(ml/ms-1):+.2f}%")
PY
echo "raw: $out"
