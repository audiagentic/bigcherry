#!/bin/bash
# RAD25: find the wait length at which a blocking HIP event wait actually sleeps (the runtime spins first). Runs
# rad25-event-wake.sh at long request periods with the blocking variants; lane CPU below 100% marks the sleep.
# Usage (from a checkout, e.g. through `bigcherry slice lab`): rad25-spin-window.sh
exec bash tools/lab/radiance/rad25-event-wake.sh 2 2000 "150 300 500 1000 2000" "yield blockdev blockonly block"
