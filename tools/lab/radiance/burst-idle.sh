#!/bin/bash
# RAD20 Option A gate: helper GPUs idle between inference bursts. Builds the moe-remote-handoff experiment, starts the
# production Flash-Next layout, samples board power and GFX activity on each helper GPU while loaded and idle, serves
# single, concurrent, back-to-back and long-prefill requests (each must answer), samples idle again, then stops.
# Gate: every request answered, clean stop, and each helper's idle median power <= RAD20_IDLE_WATTS (default 15) in
# both idle windows. The helper GPUs come from the environment: RAD20_HELPER_BDFS="<pci> <pci> ..." (required).
# Usage: RAD20_HELPER_BDFS=... burst-idle.sh [idle seconds=30]
set -u
IDLE=${1:-30}
BDFS=${RAD20_HELPER_BDFS:?set RAD20_HELPER_BDFS to the helper GPUs PCI addresses}
LIMIT=${RAD20_IDLE_WATTS:-15}
set --   # the env script reads its own arguments
source tools/env/bigcherry-env.sh >/dev/null
export ROCM_PATH=${ROCM_PATH:-/opt/rocm-7.2.4}
OUT=$BC_WORK_ROOT/runs/rad20-burst-idle-$(date +%m%d%H%M); mkdir -p $OUT
PYTHONPATH=tools timeout 5400 python3 -m bigcherry build --lane radiance-lab:stock:linux-radiance \
    --experiment moe-remote-handoff > $OUT/build.log 2>&1
plan=$(grep -oE 'ok build_plan_id=[0-9a-f]+' $OUT/build.log | tail -1 | cut -d= -f2)
[ -n "$plan" ] || { grep -E ' error:|FAILED' $OUT/build.log | sort -u | head; exit 1; }
B=$(ls -d $BC_WORK_ROOT/builds/*/$plan)
SMI=""
for bdf in $BDFS; do
    id=$(amd-smi list 2>/dev/null | awk -v b="${bdf,,}" '/^GPU:/{g=$2} tolower($0) ~ b {print g; exit}')
    [[ "$id" =~ ^[0-9]+$ ]] || { echo "GPU $bdf not found"; exit 1; }
    SMI="$SMI $id"
done
echo "t,label,gpu,watts,gfx_pct" > $OUT/samples.csv
sample() {   # $1 label, $2 seconds
    local i g
    for i in $(seq "$2"); do
        for g in $SMI; do
            amd-smi metric -g "$g" -p -u 2>/dev/null | awk -v l="$1" -v g="$g" -v t="$(date +%s.%N)" \
                '/SOCKET_POWER/{p=$2} /GFX_ACTIVITY/{a=$2} END{print t "," l "," g "," p "," a}' >> $OUT/samples.csv
        done
        sleep 1
    done
}
L=tools/lab/radiance/launch-flashnext-best.sh
R3_MOE_WORKER_BURST_TRACE=1 bash $L start --build $B > $OUT/launch.log 2>&1 || { tail $OUT/launch.log; bash $L stop; exit 1; }
URL=http://127.0.0.1:36080/v1/completions
FAILED=0
req() {
    local r
    r=$(curl -s -m 300 $URL -d "{\"prompt\":\"$1\",\"max_tokens\":$2,\"temperature\":0}")
    echo "$r" | python3 -c "import json,sys; t=json.load(sys.stdin)['choices'][0]['text']; assert t.strip()" 2>/dev/null \
        || { echo "FAIL request: $1 -> ${r:0:200}"; FAILED=1; }
}
sleep 5; sample idle_loaded "$IDLE"
req "Tell me about lighthouses." 128
for i in 1 2 3 4; do req "Story number $i about a ship." 96 & done; wait
req "One." 32; req "Two." 32; req "Three." 32
LONG=$(python3 -c "print(' '.join(['The quick brown fox jumps over the lazy dog.']*400))")
req "$LONG Summarise." 32
sleep 5; sample idle_after "$IDLE"
req "Last one after idle." 32
bash $L stop > $OUT/stop.log 2>&1 || { echo "FAIL stop"; FAILED=1; }
python3 - $OUT "$LIMIT" "$FAILED" <<'PY' | tee $OUT/summary.txt
import csv, json, statistics, sys, pathlib
out, limit, failed = pathlib.Path(sys.argv[1]), float(sys.argv[2]), sys.argv[3] == "1"
rows = list(csv.DictReader(open(out / "samples.csv")))
res, ok = {}, not failed
for label in ("idle_loaded", "idle_after"):
    for g in sorted({r["gpu"] for r in rows}):
        w = [float(r["watts"]) for r in rows if r["label"] == label and r["gpu"] == g and r["watts"]]
        a = [float(r["gfx_pct"]) for r in rows if r["label"] == label and r["gpu"] == g and r["gfx_pct"]]
        if not w: ok = False; print(f"FAIL no samples {label} gpu {g}"); continue
        m = statistics.median(w)
        res[f"{label}/gpu{g}"] = {"watts_median": m, "gfx_pct_median": statistics.median(a) if a else None, "n": len(w)}
        print(f"{label:12s} gpu {g}: median {m:6.1f} W  GFX {res[f'{label}/gpu{g}']['gfx_pct_median']} %  ({len(w)} samples)")
        if m > limit: ok = False
res["requests_failed"] = failed
json.dump(res, open(out / "results.json", "w"), indent=1)
print("IDLE_GATE", "PASS" if ok else "FAIL")
PY
echo "OUT $OUT"; echo BURST_IDLE_DONE
