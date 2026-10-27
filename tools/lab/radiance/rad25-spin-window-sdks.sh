#!/bin/bash
# RAD25: does any installed ROCm SDK sleep sooner in a blocked HIP wait? Runs the rad25 spin-window sweep (blocking
# variants at 0.15-2 ms between completions) once per ROCm SDK declared for this host in the environment settings
# (rocm-sdks), building and running the harness against each SDK's hipcc and libraries.
# Usage (from a checkout, e.g. through `bigcherry slice lab`): rad25-spin-window-sdks.sh
set -u
set --   # the env script reads its own arguments
source tools/env/bigcherry-env.sh >/dev/null
SDKS=$(PYTHONPATH=tools python3 -c '
import os
from bigcherry.core import environment as e
h = e.load_default().host(None)
for ver, root in h.rocm_sdks:
    if not os.path.isabs(root): root = os.path.join(h.repo_root, root)
    if os.path.exists(os.path.join(root, "bin", "hipcc")): print(f"{ver}={root}")
')
[ -n "$SDKS" ] || { echo "environment: no usable rocm-sdks"; exit 1; }
for sdk in $SDKS; do
    ver=${sdk%%=*}; root=${sdk#*=}
    echo "== ROCm $ver ($root)"
    ROCM_PATH=$root LD_LIBRARY_PATH=$root/lib:$root/lib64:${LD_LIBRARY_PATH:-} \
        bash tools/lab/radiance/rad25-event-wake.sh 1 2000 "300 500 1000 2000" "yield blockdev" | grep -E "^period|^OUT"
done
echo RAD25_SDKS_DONE
