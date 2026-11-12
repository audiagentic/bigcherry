#!/bin/bash
# Sourced by radiance lab scripts: host values come from the environment settings (config/environment.local.toml,
# BIGCHERRY_ENVIRONMENT), never from literals in a script.
#   ROCM_PATH   the SDK the radiance lane's stack names (a caller value that differs is refused)
#   which card does what: `bigcherry layout show NAME` (the layouts in BIGCHERRY_LAYOUTS against the inventory), not this file
#   BIGCHERRY_LAYOUTS  the layout definitions bc-serve reads (files or directories, ':'-separated); default the
#               repository's example directory engines/radiance/layouts/
_LAB_TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
_SDK=$(PYTHONPATH="$_LAB_TOOLS" python3 -c '
from pathlib import Path
from bigcherry.campaign.build import lane_sdk_root
print(lane_sdk_root(Path(__import__("sys").argv[1]), "linux-radiance"))' "$_LAB_TOOLS/../engines/radiance/recipes.toml") \
  || { echo "lab-env: the radiance lane's ROCm SDK (stack in engines/radiance/recipes.toml) is not usable" >&2; return 1 2>/dev/null || exit 1; }
if [ -n "${ROCM_PATH:-}" ] && [ "$(readlink -f "$ROCM_PATH")" != "$_SDK" ]; then
  echo "lab-env: ROCM_PATH=$ROCM_PATH is not the SDK the radiance lane names ($_SDK); unset it" >&2
  return 1 2>/dev/null || exit 1
fi
ROCM_PATH=$_SDK
export ROCM_PATH
export BIGCHERRY_LAYOUTS=${BIGCHERRY_LAYOUTS:-$_LAB_TOOLS/../engines/radiance/layouts}
