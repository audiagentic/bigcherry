#!/bin/bash
# RAD20 helper liveness, normal-path check: the watcher must cost nothing. Runs the idle-power gate (burst-idle.sh)
# and the Flash-Next speed A/B against origin/main (rad20-burst-ab.sh, 4 pairs) for this checkout. The devices come from
# the host's environment settings (device inventory): the gfx1201 card is the engine, the gfx1100 cards the helpers.
# Usage (from a checkout, e.g. through `bigcherry slice lab`): rad20-liveness-check.sh
set -u
set --   # the env script reads its own arguments
source tools/env/bigcherry-env.sh >/dev/null
read -r R9700_BDF XTX_BDFS <<<"$(PYTHONPATH=tools python3 -c '
from bigcherry.core import environment as e
h = e.load_default().host(None)
engine = [d.locator for d in h.devices_for_arch("gfx1201")]
helpers = [d.locator for d in h.devices_for_arch("gfx1100")]
assert len(engine) == 1 and helpers, (engine, helpers)
print(engine[0], " ".join(helpers))
')" || { echo "device inventory: no engine/helper GPUs"; exit 1; }
[ -n "$XTX_BDFS" ] || { echo "device inventory: no helper GPUs"; exit 1; }
echo "engine $R9700_BDF helpers $XTX_BDFS"
RAD20_HELPER_BDFS="$XTX_BDFS" bash tools/lab/radiance/burst-idle.sh 30; echo "IDLE_EXIT $?"
R9700_BDF=$R9700_BDF XTX_BDFS="$XTX_BDFS" bash tools/lab/radiance/rad20-burst-ab.sh 4 tools/lab/radiance/rad20-burst-ab.json
echo "AB_EXIT $?"; echo LIVENESS_CHECK_DONE
