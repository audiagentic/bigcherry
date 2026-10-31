#!/bin/bash
# Sourced by radiance lab scripts: host values come from the environment settings (config/environment.local.toml,
# BIGCHERRY_ENVIRONMENT), never from literals in a script.
#   ROCM_PATH   keeps the caller's value, else the host's configured rocm-root
#   lab_device_ordinals   prints "<engine ordinal> <helper ordinals...>": the gfx1201 device index, then the gfx1100 indexes
_LAB_TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
if [ -z "${ROCM_PATH:-}" ]; then
  ROCM_PATH=$(PYTHONPATH="$_LAB_TOOLS" python3 -c '
from bigcherry.core import environment as e
print(e.load_default().host(None).rocm_root)') || ROCM_PATH=
  [ -n "$ROCM_PATH" ] || { echo "lab-env: no rocm-root in the environment settings; set ROCM_PATH" >&2; return 1 2>/dev/null || exit 1; }
fi
export ROCM_PATH
lab_device_ordinals() {
  PYTHONPATH="$_LAB_TOOLS" python3 -c '
from bigcherry.core import environment as e
h = e.load_default().host(None)
engine = [d.index for d in h.devices_for_arch("gfx1201")]
helpers = [d.index for d in h.devices_for_arch("gfx1100")]
assert len(engine) == 1 and helpers, (engine, helpers)
print(engine[0], *helpers)'
}
#   lab_device_ordinals_for ARCH   prints the indexes of the host's devices of that architecture (empty when there are none)
lab_device_ordinals_for() {
  PYTHONPATH="$_LAB_TOOLS" python3 -c '
import sys
from bigcherry.core import environment as e
h = e.load_default().host(None)
print(*[d.index for d in h.devices_for_arch(sys.argv[1])])' "$1"
}
