#!/bin/bash
# PWR02 stability soak: pwr01-baseline.sh with 10-minute phases (idle, decode, prefill, idle), so a power setting runs
# under sustained decode and prefill load; the decode text hashes and the kernel log are checked by the caller.
# Usage (from a checkout, e.g. through `bigcherry slice lab`): pwr01-soak.sh
exec bash tools/lab/radiance/pwr01-baseline.sh 600
