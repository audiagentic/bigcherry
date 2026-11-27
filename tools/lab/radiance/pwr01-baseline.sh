#!/bin/bash
# PWR01 whole-system power baseline for the production Flash-Next layout: board power of every GPU in the host's device
# inventory (amd-smi) and CPU package power (RAPL energy counters, readable by the video group through a udev rule) per
# phase: idle after load, single-stream decode, back-to-back long prefills, idle again. Decode and prefill tokens are
# counted from the responses, so the summary also gives joules per token.
# Usage (from a checkout, e.g. through `bigcherry slice lab`): pwr01-baseline.sh [seconds per phase=40]
set -u
PHASE_S=${1:-40}
set --   # the env script reads its own arguments
source tools/env/bigcherry-env.sh >/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/lab-env.sh"
OUT=$BC_WORK_ROOT/runs/pwr01-baseline-$(date +%m%d%H%M); mkdir -p $OUT
git log -1 --format='commit %H %s' > $OUT/summary.txt
RAPL=/sys/class/powercap/intel-rapl:0/energy_uj
# Without the counter (no PWR01 udev rule, or a container that masks powercap) the CPU package is left out: GPU power
# and the decode/prefill rates are still measured.
[ -r $RAPL ] || { echo "CPU package energy not readable ($RAPL): CPU package power left out" | tee -a $OUT/summary.txt; RAPL=""; }
# Every GPU in the inventory: "index:locator:arch" -> amd-smi ids.
GPUS=$(PYTHONPATH=tools python3 -c '
from bigcherry.core import environment as e
print(" ".join(f"{d.locator}:{d.arch}" for d in e.load_default().host(None).devices if d.locator))
')
SMI=""; LABEL=""
for g in $GPUS; do
    bdf=${g%:*}; arch=${g##*:}
    id=$(amd-smi list 2>/dev/null | awk -v b="${bdf,,}" '/^GPU:/{n=$2} tolower($0) ~ b {print n; exit}')
    [[ "$id" =~ ^[0-9]+$ ]] || { echo "GPU $bdf not found"; exit 1; }
    SMI="$SMI $id"; LABEL="$LABEL $id=$arch@$bdf"
done
echo "gpus$LABEL" >> $OUT/summary.txt
source tools/lab/radiance/lab-build.sh
B=$(rad_build) || exit 1
echo "build $B" >> $OUT/summary.txt
# Keep the original 36080 port and the measured build pinned to this arm.
BC_SERVE="$B/radiance_home/bin/bc-serve"
[ -x "$BC_SERVE" ] || { echo "PWR01: native bc-serve missing from $B" >&2; exit 2; }
[ -n "${BC_MODEL_ROOT:-}" ] || { echo "PWR01: BC_MODEL_ROOT is required" >&2; exit 2; }
NATIVE_RUN="$OUT/session"
# The arm's own YAML carries the port and the power profile (x-bigcherry.power.profile): a managed start takes no --set and
# no profile flag.
mkdir -p "$OUT"
python3 tools/lab/radiance/arm_yaml.py --base "$B/configs/flashnext-4gpu.yaml" --out "$OUT/native-input.yaml" \
  --set server.port=36080 --set "x-bigcherry.power.profile=${PWR01_POWER_PROFILE:-stock}" || exit 2
NATIVE_ARGS=(--run-dir "$NATIVE_RUN" --engine-binary "$B/bin/radiance" --model-root "$BC_MODEL_ROOT"
             --native-config "$OUT/native-input.yaml" --work-root "$BC_WORK_ROOT" --allow-power-apply)
# Native power application requires the operator's lease and per-card
# qualification; unlike the legacy implicit mode, it cannot silently apply
# device settings without a durable receipt and restore proof.
# A freshly installed bundle has its own verified RSC12 configs. A
# development build takes the source catalogue explicitly.
[ -f "$B/configs/flashnext.toml" ] || NATIVE_ARGS+=(--configs "$PWD/engines/radiance/configs")
[ -s "$BC_WORK_ROOT/live-policy-calibration.toml" ] &&
  NATIVE_ARGS+=(--calibration "$BC_WORK_ROOT/live-policy-calibration.toml")
URL=http://127.0.0.1:36080/v1/completions
echo "t,phase,device,watts" > $OUT/samples.csv
sampler() {   # one sample a second: every GPU's board power and the CPU package power over the last second
    local e0 e1 t0 t1 max
    [ -n "$RAPL" ] && { max=$(cat /sys/class/powercap/intel-rapl:0/max_energy_range_uj); e0=$(cat $RAPL); t0=$(date +%s.%N); }
    while [ -f $OUT/.sampling ]; do
        sleep 1
        local ph; ph=$(cat $OUT/.phase)
        for g in $SMI; do
            amd-smi metric -g $g -p 2>/dev/null | awk -v t="$(date +%s.%N)" -v p="$ph" -v g="gpu$g" \
                '/SOCKET_POWER/{print t "," p "," g "," $2}' >> $OUT/samples.csv
        done
        [ -n "$RAPL" ] || continue
        e1=$(cat $RAPL); t1=$(date +%s.%N)
        python3 -c "import sys; e0,e1,t0,t1,m=map(float,sys.argv[1:6]); d=e1-e0 if e1>=e0 else e1+m-e0; print(f'{t1},{sys.argv[6]},cpu_pkg,{d/1e6/(t1-t0):.1f}')" \
            $e0 $e1 $t0 $t1 $max "$ph" >> $OUT/samples.csv
        e0=$e1; t0=$t1
    done
}
phase() { echo "$1" > $OUT/.phase; }
# Session startup and shutdown remain inside this foreground procedure's
# lease. Stop waits for every child and verified power restoration; no PID kill.
OWNER=0
SP=
stop_native() {
    local failed=0
    rm -f "$OUT/.sampling"
    if [ -n "$SP" ]; then wait "$SP" || failed=1; SP=; fi
    if [ "$OWNER" = 1 ]; then
        "$BC_SERVE" stop --run-dir "$NATIVE_RUN" > "$OUT/stop.log" 2>&1 || failed=1
        wait "$OWNER_PID" || failed=1
        OWNER=0
    fi
    return "$failed"
}
trap 'stop_native' EXIT
trap 'stop_native; exit 130' INT TERM HUP
"$BC_SERVE" serve flashnext-4gpu "${NATIVE_ARGS[@]}" > "$OUT/launch.log" 2>&1 &
OWNER_PID=$!
OWNER=1
ready=0
until=$((SECONDS + 2700))
while [ "$SECONDS" -lt "$until" ]; do
    if "$BC_SERVE" status --run-dir "$NATIVE_RUN" --json 2>/dev/null |
        python3 -c 'import json,sys
try:
    assert json.load(sys.stdin).get("state") == "running"
except (ValueError,AssertionError):
    sys.exit(1)' 2>/dev/null; then
        ready=1; break
    fi
    kill -0 "$OWNER_PID" 2>/dev/null || break
    sleep 1
done
[ "$ready" = 1 ] || { echo "PWR01: native owner did not reach running state" >&2; tail "$OUT/launch.log"; exit 1; }
phase idle_loaded; touch "$OUT/.sampling"; sampler & SP=$!
sleep 5; sleep $PHASE_S
tokens() {   # $1 prompt, $2 max_tokens: prints completion and prompt token counts
    curl -s -m 600 $URL -d "$(python3 -c 'import json,sys; print(json.dumps({"prompt":sys.argv[1],"max_tokens":int(sys.argv[2]),"temperature":0}))' "$1" "$2")" |
        python3 -c '
import hashlib, json, sys
d = json.load(sys.stdin); u = d.get("usage", {})
# Greedy output must not change with a power setting: every completion text is hashed into texts.txt.
text = (d.get("choices") or [{}])[0].get("text", "")
open(sys.argv[1], "a").write(hashlib.sha256(text.encode()).hexdigest()[:16] + "\n")
print(u.get("completion_tokens", 0), u.get("prompt_tokens", 0))' $OUT/texts-$(cat $OUT/.phase).txt
}
run_phase() {   # $1 name, $2 prompt, $3 max_tokens, $4 which count (1 completion, 2 prompt)
    phase $1
    local end=$(( $(date +%s) + PHASE_S )) n=0 t0=$(date +%s.%N)
    while [ $(date +%s) -lt $end ]; do
        # Prefill: a different first sentence each request, so the prefix cache cannot skip the work. Decode: the same
        # prompt every time, so its greedy text is a fixed reference across power settings.
        local prompt="$2"; [ $1 = prefill ] && prompt="Request $RANDOM-$n-$(date +%N). $2"
        read -r c p < <(tokens "$prompt" $3)
        [ "$4" = 1 ] && n=$((n + c)) || n=$((n + p))
    done
    echo "$1 tokens $n seconds $(python3 -c "import sys; print(round(float(sys.argv[2])-float(sys.argv[1]),1))" $t0 $(date +%s.%N))" >> $OUT/tokens.txt
}
run_phase decode "Write a long story about a lighthouse keeper and the sea." 1024 1
LONG=$(python3 -c "print(' '.join(['The quick brown fox jumps over the lazy dog.']*800))")
run_phase prefill "$LONG Summarise." 1 2
phase idle_after; sleep 5; sleep $PHASE_S
stop_native || { echo "PWR01: native stop did not verify every child exit" >&2; exit 1; }
trap - EXIT INT TERM HUP
python3 - $OUT <<'PY' | tee -a $OUT/summary.txt
import csv, statistics, sys, pathlib, re
out = pathlib.Path(sys.argv[1])
rows = list(csv.DictReader(open(out / "samples.csv")))
tok = {l.split()[0]: (int(l.split()[2]), float(l.split()[4])) for l in open(out / "tokens.txt")}
phases = ["idle_loaded", "decode", "prefill", "idle_after"]
devices = sorted({r["device"] for r in rows}, key=lambda d: (d != "cpu_pkg", d))
print("median watts per phase")
print(f"{'device':10s}" + "".join(f"{p:>13s}" for p in phases))
total = {p: 0.0 for p in phases}
for d in devices:
    line = f"{d:10s}"
    for p in phases:
        w = [float(r["watts"]) for r in rows if r["device"] == d and r["phase"] == p and r["watts"]]
        m = statistics.median(w) if w else float("nan")
        total[p] += m if w else 0
        line += f"{m:13.1f}"
    print(line)
print(f"{'total':10s}" + "".join(f"{total[p]:13.1f}" for p in phases))
for p in ("decode", "prefill"):
    if p in tok and tok[p][0]:
        n, s = tok[p]
        print(f"{p}: {n} tokens in {s} s = {n/s:.1f} tok/s, {total[p]*s/n:.3f} J/token (all devices{' + CPU package' if 'cpu_pkg' in devices else ''})")
t = out / "texts-decode.txt"
if t.exists():
    h = [l.strip() for l in open(t) if l.strip()]
    print(f"decode text: {len(set(h))} distinct of {len(h)} responses, reference {h[0] if h else '-'}")
PY
echo "OUT $OUT"; echo PWR01_DONE
