#!/bin/bash
# Reproducer runner for rad20-mes-idle.cpp (AMD report: pending AQL BARRIER_AND keeps gfx1100 at 100% GFX / ~95 W).
# Records host/GPU/firmware/ROCm details, then runs baseline, empty, drained, pending (+ released) on ONE GPU while
# amd-smi samples board power, GFX activity and GFX clock once a second; writes samples.csv, results.json and the
# kernel log lines from the run. Changes no kernel, driver or GPU setting.
# The GPU comes from the environment: RAD20_GPU_BDF=<PCI address of the card under test> (required, no default).
# Usage: RAD20_GPU_BDF=... rad20-mes-idle.sh [hold seconds=60] [output dir]
set -u
HOLD=${1:-60}
BDF=${RAD20_GPU_BDF:?set RAD20_GPU_BDF to the PCI address of the GPU under test}
OUT=${2:-./rad20-mes-idle-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"
HERE=$(cd "$(dirname "$0")" && pwd)
HIPCC=${HIPCC:-hipcc}
"$HIPCC" -O2 "$HERE/rad20-mes-idle.cpp" -o "$OUT/rad20-mes-idle" -lhsa-runtime64 || { echo "build failed"; exit 1; }

# ROCr index = position among GPU agents in KFD order, which rocm-smi --showbus lists; amd-smi id from its own list.
ROCR=$(rocm-smi --showbus 2>/dev/null | grep -i "PCI Bus: *$BDF" | grep -oE '^GPU\[[0-9]+\]' | grep -oE '[0-9]+' | head -1)
SMI=$(amd-smi list 2>/dev/null | awk -v b="${BDF,,}" '/^GPU:/{g=$2} tolower($0) ~ b {print g; exit}')
[[ "$ROCR" =~ ^[0-9]+$ && "$SMI" =~ ^[0-9]+$ ]] || { echo "GPU $BDF not found (rocm-smi index '$ROCR', amd-smi id '$SMI')"; exit 1; }
D=/sys/bus/pci/devices/$BDF
{
    echo "date: $(date -Is)"
    echo "kernel: $(uname -r)"
    echo "gpu: $BDF rocr=$ROCR amd-smi=$SMI $(rocm-smi --showproductname 2>/dev/null | grep -i "card series" | sed -n "$((ROCR + 1))p" | cut -d: -f3-)"
    for f in mes_fw_version mes_kiq_fw_version mec_fw_version me_fw_version pfp_fw_version rlc_fw_version; do
        echo "$f: $(cat $D/fw_version/$f 2>/dev/null)"
    done
    echo "rocm: $(cat ${ROCM_PATH:-/opt/rocm}/.info/version 2>/dev/null)"
    echo "amdgpu module params: sched_policy=$(cat /sys/module/amdgpu/parameters/sched_policy 2>/dev/null) mes=$(cat /sys/module/amdgpu/parameters/mes 2>/dev/null)"
    echo "linux-firmware: $(dpkg-query -W -f='${Version}' linux-firmware 2>/dev/null || rpm -q linux-firmware 2>/dev/null)"
} | tee "$OUT/environment.txt"

echo "t,label,watts,gfx_pct,gfx_mhz" > "$OUT/samples.csv"
sample() {   # $1 label, $2 seconds
    local i
    for i in $(seq "$2"); do
        amd-smi metric -g "$SMI" -p -u -c 2>/dev/null | awk -v l="$1" -v t="$(date +%s.%N)" \
            '/SOCKET_POWER/{p=$2} /GFX_ACTIVITY/{a=$2} /GFX_0:/{g=1} g&&/CLK:/{c=$2; g=0} END{print t "," l "," p "," a "," c}' >> "$OUT/samples.csv"
        sleep 1
    done
}
run_mode() {   # $1 mode
    ROCR_VISIBLE_DEVICES=$ROCR timeout $((HOLD + 60)) "$OUT/rad20-mes-idle" "$1" "$HOLD" > "$OUT/$1.log" 2>&1 &
    local pid=$!
    sleep 3
    sample "$1" $((HOLD - 6))
    [ "$1" = pending ] && { while ! grep -q "PHASE released" "$OUT/$1.log" && kill -0 $pid 2>/dev/null; do sleep 0.2; done; sample released 4; }
    wait $pid; local rc=$?
    echo "$1 rc=$rc $(grep -E 'PASS|FAIL' "$OUT/$1.log")"
    return $rc
}
START=$(date +%s)
FAILED=0
sample baseline 20
for m in empty drained pending; do run_mode $m || FAILED=1; sleep 10; done
sample baseline_end 20
journalctl -k --since "@$START" 2>/dev/null | grep -i -E "amdgpu|kfd" > "$OUT/kernel.log"

python3 - "$OUT" "$FAILED" <<'PY'
import csv, json, statistics, sys, pathlib
out, failed = pathlib.Path(sys.argv[1]), sys.argv[2] == "1"
rows = list(csv.DictReader(open(out / "samples.csv")))
res = {}
for label in dict.fromkeys(r["label"] for r in rows):
    w = sorted(float(r["watts"]) for r in rows if r["label"] == label and r["watts"])
    a = [float(r["gfx_pct"]) for r in rows if r["label"] == label and r["gfx_pct"]]
    if not w: continue
    res[label] = {"samples": len(w), "watts_median": statistics.median(w), "watts_p95": w[int(0.95 * (len(w) - 1))],
                  "gfx_pct_median": statistics.median(a) if a else None}
    print(f"{label:13s} median {res[label]['watts_median']:6.1f} W  p95 {res[label]['watts_p95']:6.1f} W  "
          f"GFX {res[label]['gfx_pct_median']} %  ({len(w)} samples)")
res["errors_in_kernel_log"] = sum(1 for l in open(out / "kernel.log") if any(k in l.lower() for k in ("error", "fail", "timeout", "hang")))
res["harness_failed"] = failed
json.dump(res, open(out / "results.json", "w"), indent=1)
PY
echo "OUT $OUT"
exit $FAILED
