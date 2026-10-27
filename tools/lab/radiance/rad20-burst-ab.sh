#!/bin/bash
# RAD20 Option A speed A/B: builds the moe-remote-handoff experiment from origin/main (in a worktree under this
# checkout's worktrees/) and from this branch, then runs rad20-burst-ab.json through ab-benchmark (paired, order-balanced).
# Devices come from the environment: R9700_BDF (engine GPU PCI address) and XTX_BDFS (helper GPU PCI addresses); the
# ROCr indices are looked up from them.
# Usage: R9700_BDF=... XTX_BDFS="... ..." rad20-burst-ab.sh [pairs=4] [server config=rad20-burst-ab.json; rad20-burst-streams-ab.json for concurrency]
set -u
PAIRS=${1:-4}
CONFIG=${2:-tools/lab/radiance/rad20-burst-ab.json}
: "${R9700_BDF:?set R9700_BDF}" "${XTX_BDFS:?set XTX_BDFS}"
set --   # the env script reads its own arguments
source tools/env/bigcherry-env.sh >/dev/null
export ROCM_PATH=${ROCM_PATH:-/opt/rocm-7.2.4}
OUT=$BC_WORK_ROOT/runs/rad20-burst-ab-$(date +%m%d%H%M); mkdir -p $OUT
rocr() { rocm-smi --showbus 2>/dev/null | grep -i "PCI Bus: *$1" | grep -oE '^GPU\[[0-9]+\]' | grep -oE '[0-9]+' | head -1; }
export R9700_ROCR=$(rocr "$R9700_BDF")
XTX_DEVICES=""; for b in $XTX_BDFS; do XTX_DEVICES="$XTX_DEVICES $(rocr "$b")"; done
export XTX_DEVICES=${XTX_DEVICES# }
[[ "$R9700_ROCR" =~ ^[0-9]+$ && "$XTX_DEVICES" =~ ^[0-9]+(\ [0-9]+)*$ ]] || { echo "device lookup failed: '$R9700_ROCR' '$XTX_DEVICES'"; exit 1; }
build() {   # $1 label, $2 checkout (built, or reused when its radiance source tree matches: lab-build.sh)
    (cd "$2" && source "$OLDPWD/tools/lab/radiance/lab-build.sh" && rad_build $OUT/build-$1.log)
}
MAIN_WT=$(git rev-parse --path-format=absolute --git-common-dir)/../worktrees/main-baseline
git fetch -q origin
[ -d "$MAIN_WT" ] || git worktree add -q --detach "$MAIN_WT" origin/main
git -C "$MAIN_WT" checkout -q --detach origin/main
echo "main $(git -C "$MAIN_WT" log --oneline -1)" | tee $OUT/arms.txt
echo "burst $(git log --oneline -1)" | tee -a $OUT/arms.txt
B_MAIN=$(build main "$MAIN_WT") || exit 1
B_BURST=$(build burst "$PWD") || exit 1
export RAD_MAIN_BIN=$B_MAIN/bin/radiance RAD_MAIN_HOME=$B_MAIN/radiance_home
export RAD_BURST_BIN=$B_BURST/bin/radiance RAD_BURST_HOME=$B_BURST/radiance_home
export R9700_BDF
PYTHONPATH=tools python3 -m bigcherry ab-benchmark --server-config "$CONFIG" \
    --pairs "$PAIRS" --output $OUT/ab 2>&1 | tee $OUT/ab.log | tail -40
echo "OUT $OUT"; echo BURST_AB_DONE
