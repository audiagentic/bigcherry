#!/bin/bash
# BCOP37 step 1: record the host as it is now (GPUs, gfx targets, PCIe links, ROCm, CPU/NUMA, RAM, swap, free VRAM)
# into <bundle>/system.txt. Read-only.
# Usage: bcop37-system.sh <bundle dir>
set -u
B=${1:?bundle dir}; mkdir -p "$B"
ROCM=${ROCM:-/mnt/vault/tmp/bc-rocm}
{
  echo "date: $(date -Is)"; echo "host: $(hostname)"; echo "kernel: $(uname -r)"; grep PRETTY /etc/os-release
  echo; echo "## CPU / NUMA"; lscpu | grep -E "Model name|Socket|Core|Thread|NUMA|^CPU\(s\)"
  echo; echo "## RAM / swap"; free -g; swapon --show; grep -E "MemTotal|MemAvailable|SwapTotal|SwapFree|Mlocked" /proc/meminfo
  echo; echo "## ROCm"; echo "ROCM=$ROCM"; cat "$ROCM"/.info/version* 2>/dev/null; "$ROCM"/bin/hipconfig --version 2>/dev/null
  cat /sys/module/amdgpu/version 2>/dev/null
  echo; echo "## GPUs (rocm-smi)"; "$ROCM"/bin/rocm-smi --showproductname --showmeminfo vram --showbus 2>/dev/null || rocm-smi --showproductname --showmeminfo vram --showbus 2>/dev/null
  echo; echo "## HIP device order and gfx target"; "$ROCM"/bin/rocminfo 2>/dev/null | grep -E "Marketing Name|Name:.*gfx|Device Type|Node:" | paste - - - - 2>/dev/null | grep -i gpu
  echo; echo "## PCIe negotiated links"
  for d in /sys/class/drm/card*/device; do
    [ -f "$d/vendor" ] && [ "$(cat "$d/vendor")" = 0x1002 ] || continue
    echo "$(basename "$(readlink -f "$d")") device=$(cat "$d/device") link=$(cat "$d/current_link_speed" 2>/dev/null) x$(cat "$d/current_link_width" 2>/dev/null) max=$(cat "$d/max_link_speed" 2>/dev/null) x$(cat "$d/max_link_width" 2>/dev/null) vram_total=$(( $(cat "$d/mem_info_vram_total") >> 20 ))MiB vram_used=$(( $(cat "$d/mem_info_vram_used") >> 20 ))MiB"
  done
  echo; echo "## P2P / RCCL"; echo "see PHA evidence in docs/planning (no P2P between discrete cards on this host; RCCL host-mediated)"
} > "$B/system.txt" 2>&1
echo "wrote $B/system.txt ($(wc -l < "$B/system.txt") lines)"
