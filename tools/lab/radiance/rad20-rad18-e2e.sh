#!/bin/bash
# RAD20 + RAD18 in one build-server session (run through `bigcherry slice lab feat/rad20-blocking-lanes ...`):
#   1. build this slice (RAD20: blocking helper lanes) and its parent commit (main: the RAD18 combine binary and
#      RAD20's baseline arm), both --experiment moe-remote-handoff; no build runs during the measurements below;
#   2. RAD20 A/B, helper-rad20-ab.json (polling vs blocking), with pidstat sampling every helper thread's CPU;
#   3. RAD18 on the main build: helper-combine-ab.json (combine off/on) and helper-combine-nofire-ab.json (off vs
#      on-but-never-fires), for the tg128 check; every cell now records MTP draft totals (cell.json "draft").
# Usage: rad20-rad18-e2e.sh [pairs=4]
set -u
PAIRS=${1:-4}
W=${BC_WORK_ROOT}
OUT=$W/runs/rad20-rad18-$(date +%m%d%H%M)
mkdir -p $OUT
export ROCM_PATH=/opt/rocm-7.2.4 PATH=/opt/rocm-7.2.4/bin:$PATH
git fetch -q origin main
HEAD=$(git rev-parse HEAD); BASE=$(git merge-base HEAD FETCH_HEAD)
echo "slice $HEAD baseline $BASE" | tee $OUT/builds.txt

build() {   # $1 label: builds the checked-out tree, prints its runtime prefix
    PYTHONPATH=tools timeout 5400 python3 -m bigcherry build --lane radiance-lab:stock:linux-radiance \
        --experiment moe-remote-handoff > $OUT/build-$1.log 2>&1
    local plan; plan=$(grep -oE 'ok build_plan_id=[0-9a-f]+' $OUT/build-$1.log | tail -1 | cut -d= -f2)
    [ -n "$plan" ] || { grep -E ' error:|FAILED' $OUT/build-$1.log | sort -u | head >&2; return 1; }
    ls -d $W/builds/*/$plan
}
NEW=$(build new) || { echo "RAD20 build failed"; exit 1; }
git checkout -q --detach $BASE
OLD=$(build base); rc=$?
git checkout -q --detach $HEAD
[ $rc -eq 0 ] || { echo "baseline build failed"; exit 1; }
for p in $NEW $OLD; do echo "$p $(sha256sum $p/radiance_home/helpers/r3_moe_worker | cut -c1-16)"; done | tee -a $OUT/builds.txt

export RAD_BIN_NEW=$NEW/bin/radiance RAD_BIN_BASE=$OLD/bin/radiance RAD_BIN=$OLD/bin/radiance \
       RAD_HOME=$OLD/radiance_home R9700_BDF=0000:09:00.0

ab() {   # $1 config, $2 output name
    PYTHONPATH=tools timeout 14400 python3 -m bigcherry ab-benchmark --server-config tools/lab/radiance/$1 \
        --output $OUT/$2 --pairs $PAIRS --settle-seconds 20 > $OUT/$2.log 2>&1
    tail -2 $OUT/$2.log
    for d in $OUT/$2/pair-*; do
        python3 - "$d/cell.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); m = c.get("metrics") or {}; dr = c.get("draft") or {}
acc = f"{dr['accepted']}/{dr['drafted']}" if dr.get("drafted") else "-"
print(sys.argv[1].split("/")[-2], c.get("position"), " ".join(f"{k}={m[k]:.1f}" for k in sorted(m) if k.endswith("_tps")),
      "draft", acc, c.get("metric_error", ""))
PY
    done
}

# 2. RAD20, with every r3_moe_worker thread's CPU sampled (pidstat -t, 5 s) for the whole run; cells run in the order
#    ab.log lists, so the n-th pair of helper PIDs is the n-th cell.
pidstat -t -h -u -C r3_moe_worker 5 > $OUT/rad20-pidstat.txt 2>&1 &
PS=$!
echo "== RAD20 polling vs blocking"; ab helper-rad20-ab.json rad20
kill $PS 2>/dev/null; wait $PS 2>/dev/null
# 3. RAD18 on the main build
echo "== RAD18 combine off/on";            ab helper-combine-ab.json rad18-combine
echo "== RAD18 combine off/never-fires";   ab helper-combine-nofire-ab.json rad18-nofire
echo "OUT $OUT"
echo RAD20_RAD18_DONE
