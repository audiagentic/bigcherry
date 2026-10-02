#!/bin/bash
# Diagnose the 27B dual-XTX prefill gap (pp4096 ~921 t/s on 2026-10-02 with healthy links vs ~1420 on
# 2026-08-26): (1) which RCCL transport the two XTXs use (NCCL_DEBUG=INFO: P2P/IPC vs SHM), (2) pp4096
# with the CPU governor at powersave vs performance (sudo -S, password on stdin; restored afterwards),
# (3) GPU clocks during the run. Usage: pp-regression-diag.sh <llama-server> <out-dir> < password
set -u
srv=$1 out=$2; mkdir -p "$out"
read -r pw
bench=$(dirname "$srv")/llama-bench
model=/mnt/data/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
run() {
  HIP_VISIBLE_DEVICES=0,1 ROCR_VISIBLE_DEVICES=0,1 "$@" "$bench" -m "$model" -sm tensor -ngl 99 -fa 1 \
    -p 4096 -n 0 -ub 512 -b 2048 -r 3 -o md
}
echo "== RCCL transport"
run env NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,P2P,SHM 2> "$out/nccl.log" | tee "$out/nccl-run.md"
grep -hE "via (P2P|SHM|NET)|P2P is disabled|p2p" "$out/nccl.log" | sort | uniq -c | head -20
gov0=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
for g in powersave performance; do
  echo "$pw" | sudo -S -p "" cpupower frequency-set -g $g >/dev/null 2>&1 \
    || for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo "$pw" | sudo -S -p "" sh -c "echo $g > $c"; done
  echo "== governor $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
  ( for i in $(seq 40); do echo "$(date +%T) sclk0=$(grep '\*' /sys/bus/pci/devices/0000:03:00.0/pp_dpm_sclk | cut -d: -f2) mclk0=$(grep '\*' /sys/bus/pci/devices/0000:03:00.0/pp_dpm_mclk | cut -d: -f2) fclk0=$(grep '\*' /sys/bus/pci/devices/0000:03:00.0/pp_dpm_fclk 2>/dev/null | cut -d: -f2)"; sleep 1; done > "$out/clocks-$g.txt" ) &
  run 2> "$out/bench-$g.stderr" | tee "$out/bench-$g.md"
  wait
done
for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo "$pw" | sudo -S -p "" sh -c "echo $gov0 > $c"; done
echo "governor restored to $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
echo DIAG_DONE
