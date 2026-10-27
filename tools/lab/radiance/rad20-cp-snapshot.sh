#!/bin/bash
# RAD20: read-only CP/GRBM register snapshots (AMD UMR) on one GPU through the rad20-mes-idle phases: idle, empty queue,
# pending BARRIER_AND at +2 / +10 / +25 s, released, destroyed. Shows which block stays busy while a barrier waits.
# Needs: UMR=<path to the umr binary> (built from https://gitlab.freedesktop.org/tomstdenis/umr; reads registers through
# debugfs, so it runs under sudo), RAD20_GPU_BDF=<PCI address of the GPU>, and the rad20-mes-idle binary next to this
# script or RAD20_MES_IDLE=<path>. Writes nothing to the GPU.
# Usage: UMR=... RAD20_GPU_BDF=... rad20-cp-snapshot.sh [output file]
set -u
U=${UMR:?set UMR to the umr binary}
BDF=${RAD20_GPU_BDF:?set RAD20_GPU_BDF to the PCI address of the GPU under test}
HERE=$(cd "$(dirname "$0")" && pwd)
BIN=${RAD20_MES_IDLE:-$HERE/rad20-mes-idle}
OUT=${1:-./rad20-cp-snapshot-$(date +%Y%m%d-%H%M%S).txt}
ROCR=$(rocm-smi --showbus 2>/dev/null | grep -i "PCI Bus: *$BDF" | grep -oE '^GPU\[[0-9]+\]' | grep -oE '[0-9]+' | head -1)
[[ "$ROCR" =~ ^[0-9]+$ ]] || { echo "GPU $BDF not found"; exit 1; }
REGS="GRBM_STATUS GRBM_STATUS2 GRBM_STATUS3 CP_STAT CP_BUSY_STAT CP_CPC_STATUS CP_CPC_BUSY_STAT CP_CPC_BUSY_STAT2 CP_CPC_STALLED_STAT1 CP_STALLED_STAT1 RLC_STAT"
snap() {
    echo "== $1"
    for r in $REGS; do sudo "$U" --by-pci "$BDF" --read "*.*.$r" 2>&1 | grep -E "\.reg$r => "; done
}
LOG=$(mktemp)
{
    snap S0_idle
    ROCR_VISIBLE_DEVICES=$ROCR timeout 60 "$BIN" empty 12 > "$LOG.empty" 2>&1 &
    sleep 5; snap S1_empty; wait
    ROCR_VISIBLE_DEVICES=$ROCR timeout 90 "$BIN" pending 30 > "$LOG" 2>&1 &
    sleep 3; snap S2_pending_2s
    sleep 8; snap S3_pending_10s
    sleep 14; snap S4_pending_25s
    while ! grep -q "PHASE released " "$LOG" && kill -0 %1 2>/dev/null; do sleep 0.2; done
    sleep 1; snap S5_released; wait
    snap S6_destroyed
    echo "== decoded (pending)"
    ROCR_VISIBLE_DEVICES=$ROCR timeout 60 "$BIN" pending 12 > "$LOG.bits" 2>&1 &
    sleep 4
    for r in CP_CPC_STATUS CP_CPC_BUSY_STAT CP_CPC_STALLED_STAT1 GRBM_STATUS GRBM_STATUS2 GRBM_STATUS3; do
        sudo "$U" --by-pci "$BDF" -O bits --read "*.*.$r" 2>&1 | grep -E "\.reg$r =>|==[[:space:]]+[1-9]"
    done
    wait
    grep -h -E "PASS|FAIL" "$LOG.empty" "$LOG" "$LOG.bits"
} | tee "$OUT"
rm -f "$LOG" "$LOG.empty" "$LOG.bits"
