#!/bin/bash
# RAD26 validation for this checkout: idle-power gate, Flash-Next speed A/B against origin/main (4 pairs, combine on),
# the concurrency A/B (1/2/4/6 streams, 2 pairs; the large lane also serves multi-stream decode steps above the small
# class), and the helper kill test. Devices come from the host's device inventory (environment settings).
# Usage (from a checkout, e.g. through `bigcherry slice lab`): rad26-check.sh
set -u
set --   # the env script reads its own arguments
source tools/env/bigcherry-env.sh >/dev/null
read -r R9700_BDF XTX_BDFS <<<"$(PYTHONPATH=tools python3 -c '
from bigcherry.core import environment as e
h = e.load_default().host(None)
print(h.devices_for_arch("gfx1201")[0].locator, " ".join(d.locator for d in h.devices_for_arch("gfx1100")))
')" || { echo "device inventory: no engine/helper GPUs"; exit 1; }
RAD20_HELPER_BDFS="$XTX_BDFS" bash tools/lab/radiance/burst-idle.sh 30; echo "IDLE_EXIT $?"
R9700_BDF=$R9700_BDF XTX_BDFS="$XTX_BDFS" bash tools/lab/radiance/rad20-burst-ab.sh 4 tools/lab/radiance/rad20-burst-ab.json; echo "AB_EXIT $?"
R9700_BDF=$R9700_BDF XTX_BDFS="$XTX_BDFS" bash tools/lab/radiance/rad20-burst-ab.sh 2 tools/lab/radiance/rad20-burst-streams-ab.json; echo "STREAMS_EXIT $?"
bash tools/lab/radiance/rad20-helper-kill.sh; echo "KILL_EXIT $?"
echo RAD26_CHECK_DONE
