#!/bin/bash
# RAD20: driver-level view of the helper GPUs' hardware queues with power save on and off. For each mode, starts the
# production Flash-Next layout from one build, and while a long decode runs takes KFD debugfs snapshots (kfd/mqds: queues
# each process owns; kfd/hqds: which hardware queue slots hold a queue) and UMR CP/MES status on each helper GPU; then
# the same idle after the request. Shows whether the helpers oversubscribe the GPU's compute slots and whether MES stays
# busy. Reads debugfs and registers only (sudo); changes nothing.
# Needs: RAD20_BUILD (a moe-remote-handoff build), XTX_BDFS (helper GPU PCI addresses), UMR (umr binary), sudo. The sudo
# password is read once from stdin (so a detached run can use it; it is never written anywhere).
# Usage: RAD20_BUILD=... XTX_BDFS="..." UMR=... rad20-queue-probe.sh < <password on stdin>
set -u
: "${RAD20_BUILD:?set RAD20_BUILD}" "${XTX_BDFS:?set XTX_BDFS}" "${UMR:?set UMR}"
IFS= read -r PW || PW=""
sudo() { command sudo -S -p "" "$@" <<< "$PW"; }
set --   # the env script reads its own arguments
source tools/env/bigcherry-env.sh >/dev/null
export ROCM_PATH=${ROCM_PATH:-/opt/rocm-7.2.4}
OUT=$BC_WORK_ROOT/runs/rad20-queue-probe-$(date +%m%d%H%M); mkdir -p $OUT
HERE=$(cd "$(dirname "$0")" && pwd)
L=$HERE/launch-flashnext-best.sh
URL=http://127.0.0.1:36080/v1/completions
snap() {   # $1 label: queue ownership, resident hardware slots, CP/MES status
    local f=$OUT/$MODE-$1
    sudo cat /sys/kernel/debug/kfd/mqds > $f.mqds
    sudo cat /sys/kernel/debug/kfd/hqds > $f.hqds
    for b in $XTX_BDFS; do
        for r in CP_CPC_STATUS CP_CPC_BUSY_STAT GRBM_STATUS; do
            sudo "$UMR" --by-pci "$b" --read "*.*.$r" 2>&1 | grep -E "\.reg$r => " | sed "s/^/$b /"
        done
    done > $f.regs
}
for MODE in on off; do
    v=1; [ $MODE = off ] && v=0
    R3_MOE_WORKER_POWER_SAVE=$v bash $L start --build $RAD20_BUILD > $OUT/$MODE-launch.log 2>&1 || { tail $OUT/$MODE-launch.log; bash $L stop; exit 1; }
    curl -s $URL -d '{"prompt":"Warm up.","max_tokens":16,"temperature":0}' > /dev/null
    curl -s $URL -d '{"prompt":"Write a very long story about a lighthouse keeper.","max_tokens":1500,"temperature":0}' > /dev/null &
    RQ=$!
    sleep 4; snap decode1; sleep 2; snap decode2
    wait $RQ
    sleep 5; snap idle
    bash $L stop > $OUT/$MODE-stop.log 2>&1
done
python3 - $OUT <<'PY'
import re, sys, pathlib
out = pathlib.Path(sys.argv[1])
for f in sorted(out.glob("*.hqds")):
    # a hardware slot holds a queue when its register block is not all zero
    blocks = re.split(r"\n(?=\s*Inst \d+,)", f.read_text())
    node, res = None, {}
    for b in blocks:
        m = re.search(r"gpu_id (\w+)", b)
        if m: node = m.group(1)
        if "CP Pipe" in b and re.search(r": .*[1-9a-f]", b.split("\n", 1)[-1]):
            res[node] = res.get(node, 0) + 1
    mq = f.with_suffix(".mqds").read_text()
    per = {}
    for m in re.finditer(r"Compute queue on device (\w+)", mq):
        per[m.group(1)] = per.get(m.group(1), 0) + 1
    print(f"{f.stem:14s} resident CP slots {res}   queues owned {per}")
for f in sorted(out.glob("*.regs")):
    print(f"== {f.stem}"); print(f.read_text().strip())
PY
echo "OUT $OUT"; echo QUEUE_PROBE_DONE
