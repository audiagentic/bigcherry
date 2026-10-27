#!/bin/bash
# Runner for rad20-queue-warm.cpp on one GPU: base, active and parked placeholder modes, each followed by an idle hold
# while amd-smi samples board power and GFX activity; prints the dispatch-gap and toggle results and the hold power.
# The GPU comes from the environment: RAD20_GPU_BDF=<PCI address> (required).
# Usage: RAD20_GPU_BDF=... rad20-queue-warm.sh [placeholders=10] [rounds=2000] [hold seconds=15]
set -u
P=${1:-10}; ROUNDS=${2:-2000}; HOLD=${3:-15}
BDF=${RAD20_GPU_BDF:?set RAD20_GPU_BDF to the PCI address of the GPU under test}
OUT=./rad20-queue-warm-$(date +%Y%m%d-%H%M%S); mkdir -p "$OUT"
HERE=$(cd "$(dirname "$0")" && pwd)
"${HIPCC:-hipcc}" -O2 "$HERE/rad20-queue-warm.cpp" -o "$OUT/rad20-queue-warm" -lhsa-runtime64 || { echo "build failed"; exit 1; }
ROCR=$(rocm-smi --showbus 2>/dev/null | grep -i "PCI Bus: *$BDF" | grep -oE '^GPU\[[0-9]+\]' | grep -oE '[0-9]+' | head -1)
SMI=$(amd-smi list 2>/dev/null | awk -v b="${BDF,,}" '/^GPU:/{g=$2} tolower($0) ~ b {print g; exit}')
[[ "$ROCR" =~ ^[0-9]+$ && "$SMI" =~ ^[0-9]+$ ]] || { echo "GPU $BDF not found"; exit 1; }
FAILED=0
for m in base active parked; do
    ROCR_VISIBLE_DEVICES=$ROCR timeout $((HOLD + 120)) "$OUT/rad20-queue-warm" $m "$P" "$ROUNDS" "$HOLD" > "$OUT/$m.log" 2>&1 &
    pid=$!
    while ! grep -q "PHASE hold-begin" "$OUT/$m.log" && kill -0 $pid 2>/dev/null; do sleep 0.2; done
    sleep 2
    w=(); a=()
    for i in $(seq $((HOLD - 4))); do
        read -r pw ga < <(amd-smi metric -g "$SMI" -p -u 2>/dev/null | awk '/SOCKET_POWER/{p=$2} /GFX_ACTIVITY/{g=$2} END{print p, g}')
        w+=("$pw"); a+=("$ga"); sleep 1
    done
    wait $pid || FAILED=1
    grep -E "RESULT|PASS|FAIL" "$OUT/$m.log"
    printf "%s hold: power median %s W, GFX median %s %%\n" "$m" \
        "$(printf '%s\n' "${w[@]}" | sort -n | awk '{v[NR]=$1} END{print v[int((NR+1)/2)]}')" \
        "$(printf '%s\n' "${a[@]}" | sort -n | awk '{v[NR]=$1} END{print v[int((NR+1)/2)]}')"
done
echo "OUT $OUT"
exit $FAILED
