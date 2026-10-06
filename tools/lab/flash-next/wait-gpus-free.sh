#!/bin/bash
# Block until the lab can have the GPUs: no llama-server that llama-swap started (port 42007) is running and the
# given cards hold less than LIMIT_MIB of VRAM in total. It never stops anything - a production model that is
# loaded is someone using the machine; the lab waits for it to unload.
# Usage: wait-gpus-free.sh [cards="0 1 2"]      env: LIMIT_MIB (2000), POLL_S (60)
set -u
cards=${1:-"0 1 2"}
used() {
  local total=0 c v
  for c in $cards; do
    v=$(rocm-smi --showmeminfo vram 2>/dev/null | grep "GPU\[$c\].*Total Used" | awk '{print int($NF / 1048576)}')
    total=$((total + ${v:-99999}))
  done
  echo "$total"
}
while pgrep -f "[l]lama-server --port 42007" > /dev/null || [ "$(used)" -ge "${LIMIT_MIB:-2000}" ]; do
  sleep "${POLL_S:-60}"
done
echo "GPUS_FREE $(date -Is) used=$(used) MiB on cards $cards"
