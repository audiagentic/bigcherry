#!/bin/bash
# Per-card usage while lab runs execute: every INTERVAL seconds one rocm-smi sample (GPU use %, power W, VRAM used) per
# card, tagged with the arm that is running (the output directory of the long-ctx-profile.sh / moe-copy-ab.sh process,
# last two path components; "idle" when none runs). Read-only, started beside a queue and stopped by its marker.
# Usage: gpu-usage-sampler.sh <out.csv> [stop-file-with-marker] [marker=CHAIN]     env: INTERVAL (5)
#        gpu-usage-sampler.sh --summary <out.csv>      per arm and card: mean and peak use, mean power, peak VRAM
set -u
if [ "${1:-}" = --summary ]; then
  awk -F, 'NR > 1 && $2 != "idle" { k = $2 SUBSEP $3; n[k]++; u[k] += $5; p[k] += $4; if ($5 > mu[k]) mu[k] = $5; if ($6 > mv[k]) mv[k] = $6; if ($5 >= 50) b[k]++ }
    END { for (k in n) { split(k, a, SUBSEP); printf "%-44s %s  use mean %5.1f%% peak %3d%%  busy(>=50%%) %5.1f%% of samples  power %5.1f W  vram peak %6d MiB  (%d samples)\n", a[1], a[2], u[k] / n[k], mu[k], 100 * b[k] / n[k], p[k] / n[k], mv[k], n[k] } }' "$2" | sort
  exit 0
fi
out=${1:?out.csv}; stop=${2:-}; marker=${3:-CHAIN}
echo "epoch,arm,card,power_w,use_pct,vram_mib" > "$out"
while :; do
  [ -n "$stop" ] && grep -q "^${marker}.*_DONE" "$stop" 2>/dev/null && break
  arm=$(pgrep -af "long-ctx-profile.sh|moe-copy-ab.sh" | grep -oE "/mnt/data/bigcherry-work/runs/[^ ]+" | head -1 | awk -F/ '{print $(NF-1) "/" $NF}')
  now=$(date +%s)
  rocm-smi --showuse --showmeminfo vram --showpower --csv 2>/dev/null | awk -F, -v t="$now" -v arm="${arm:-idle}" \
    '$1 ~ /^card[0-9]/ { printf "%s,%s,%s,%s,%s,%d\n", t, arm, $1, $2, $3, $5 / 1048576 }' >> "$out"
  sleep "${INTERVAL:-5}"
done
