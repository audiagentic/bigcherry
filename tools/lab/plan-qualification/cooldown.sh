#!/bin/bash
# GPU cooldown between back-to-back measurement sessions (PA35: no gap at all caused clock instability).
# Waits at least COOL_MIN seconds, then until every GPU's edge temperature is <= COOL_TEMP C, capped at
# COOL_MAX seconds. Defaults: 30 s / 55 C / 300 s. Source it and call: gpu_cooldown
gpu_cooldown() {
  local min=${COOL_MIN:-30} temp=${COOL_TEMP:-55} max=${COOL_MAX:-300} waited=0 hot
  sleep "$min"; waited=$min
  while [ "$waited" -lt "$max" ]; do
    hot=$(rocm-smi --showtemp 2>/dev/null | awk -v t="$temp" '/Sensor edge/ {if ($NF+0 > t) n++} END {print n+0}')
    [ "$hot" -eq 0 ] && break
    sleep 10; waited=$((waited + 10))
  done
  echo "cooldown ${waited}s (edge <= ${temp}C or cap ${max}s)"
}
