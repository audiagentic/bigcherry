#!/usr/bin/env bash
# RAD20 option B lab runner for rad20-barrier-wake.cpp: builds it, checks the GPUs are idle, then runs the
# single-process case and the two-process case (helper process owns the XTX queue and creates the IPC signal,
# engine process attaches it and releases) for three schedules:
#   b2b      release 100 us after arming, iterations back to back
#   pending  release 20 ms after arming: the wait sits armed on an otherwise idle XTX
#   cold     200 ms with nothing queued on the XTX before arming, release 100 us after arming
# Needs R9700_BDF and XTX_BDF (PCI addresses, e.g. 0000:09:00.0 and 0000:03:00.0). Hold the gpu lease around it:
#   agent-lease acquire gpu --ttl 30m --purpose "RAD20 barrier wake" -- tools/lab/radiance/rad20-barrier-wake.sh
# Iteration counts: N_B2B (300), N_PENDING (50), N_COLD (30). Output: $RAD20_OUT (default
# ./rad20-barrier-wake-<UTC time>) with one log per case and summary.txt; exit status 1 if any arm FAILs.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
: "${R9700_BDF:?set R9700_BDF to the R9700 PCI address, e.g. 0000:09:00.0}"
: "${XTX_BDF:?set XTX_BDF to the RX 7900 XTX PCI address, e.g. 0000:03:00.0}"
export R9700_BDF XTX_BDF
out=${RAD20_OUT:-$PWD/rad20-barrier-wake-$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$out"
bin=$out/rad20-barrier-wake

# The GPUs must be idle: no radiance engine, helper or A/B run, and no ROCm (KFD) process at all.
busy=$(ps -eo pid=,comm= | awk '$2 ~ /^(radiance|r3_moe_worker|ab-benchmark)/' || true)
if [[ -n $busy ]]; then
  echo "FAIL GPUs busy:"; echo "$busy"; exit 2
fi
if command -v rocm-smi >/dev/null; then
  kfd=$(rocm-smi --showpids 2>/dev/null |
        awk '/KFD Processes/ {f = 1; next} /^=+$/ {if (f) exit} f && $1 ~ /^[0-9]+$/' || true)
  if [[ -n $kfd ]]; then
    echo "FAIL ROCm processes on the GPUs:"; echo "$kfd"; exit 2
  fi
fi

hipcc -O2 -std=c++17 ${RAD20_HIPCC_FLAGS:-} -o "$bin" "$here/rad20-barrier-wake.cpp" -lhsa-runtime64

schedules=("b2b ${N_B2B:-300} 100 0" "pending ${N_PENDING:-50} 20000 0" "cold ${N_COLD:-30} 100 200000")
single_arms=A.gpu,A.host,B.gpu,B.host
ipc_arms=A.gpu,A.host,A.self,B.gpu,B.host,B.self
status=0
for s in "${schedules[@]}"; do
  read -r name n gap idle <<<"$s"
  args=(--n "$n" --gap-us "$gap" --idle-us "$idle")
  timeout 600 "$bin" --mode single --arms "$single_arms" "${args[@]}" >"$out/single-$name.log" 2>&1 || status=1

  shm=/rad20-barrier-wake-$$-$name
  timeout 600 "$bin" --mode helper --shm "$shm" --arms "$ipc_arms" "${args[@]}" >"$out/ipc-$name-helper.log" 2>&1 &
  hpid=$!
  timeout 600 "$bin" --mode engine --shm "$shm" --arms "$ipc_arms" "${args[@]}" >"$out/ipc-$name-engine.log" 2>&1 &
  epid=$!
  wait "$hpid" || status=1
  wait "$epid" || status=1
done

{
  echo "rad20-barrier-wake $(date -u +%FT%TZ) host=$(hostname) rev=$(git -C "$here" rev-parse --short HEAD 2>/dev/null || echo ?)"
  echo "R9700_BDF=$R9700_BDF XTX_BDF=$XTX_BDF hip=$(hipconfig --version 2>/dev/null || echo ?)"
  for f in "$out"/single-*.log "$out"/ipc-*-helper.log "$out"/ipc-*-engine.log; do
    echo "== $(basename "$f")"
    grep -E '^(PASS|FAIL|arm=|clock|helper:|engine:)' "$f" || true
  done
} | tee "$out/summary.txt"
exit "$status"
