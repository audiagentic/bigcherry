#!/bin/bash
# Run queue scripts strictly one after another (each waits for the previous to exit, vLLM restart included).
# Parallel queue scripts overlapped on 2026-10-03: one queue's end-of-run vLLM restart / servers took the R9700
# while another queue's servers loaded, so whole A/Bs OOMed. Usage: chain-serial.sh <wait-log> <queue.sh>...
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
wait_log=$1; shift
until grep -q ALL_JOBS_DONE "$wait_log" 2>/dev/null; do sleep 30; done
for q in "$@"; do
  name=$(basename "$q" .sh)
  echo "== $name $(date -Is)"
  bash "$q" > "work/runs/$name-serial.log" 2>&1
  tail -2 "work/runs/$name-serial.log"
done
echo CHAIN_DONE
