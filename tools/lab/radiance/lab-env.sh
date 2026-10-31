#!/bin/bash
# Sourced by radiance lab scripts: host values come from the environment settings (config/environment.local.toml,
# BIGCHERRY_ENVIRONMENT), never from literals in a script.
#   ROCM_PATH   keeps the caller's value, else the host's configured rocm-root
#   which card does what: `bigcherry layout show NAME` (engines/radiance/layouts.toml against the inventory), not this file
_LAB_TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
if [ -z "${ROCM_PATH:-}" ]; then
  ROCM_PATH=$(PYTHONPATH="$_LAB_TOOLS" python3 -c '
from bigcherry.core import environment as e
print(e.load_default().host(None).rocm_root)') || ROCM_PATH=
  [ -n "$ROCM_PATH" ] || { echo "lab-env: no rocm-root in the environment settings; set ROCM_PATH" >&2; return 1 2>/dev/null || exit 1; }
fi
export ROCM_PATH
