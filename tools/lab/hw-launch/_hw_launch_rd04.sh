#!/bin/bash
set -x
nohup bash /home/audumla/_hw_rd04_gfx1100.sh >/mnt/vault/tmp/bc-runs/rd04-gfx1100-contract-run2.log 2>&1 &
RD04_PID=$!
echo "RD04_PID=$RD04_PID"
sleep 3
if kill -0 "$RD04_PID" 2>/dev/null; then echo "RD04 ALIVE"; else echo "RD04 DEAD"; fi
