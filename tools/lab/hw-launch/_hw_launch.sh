#!/bin/bash
# Launcher: create run dirs and start RD13 (gfx1030) + RD26 (gfx1100) in the
# background. Kept in a file so the backgrounding is done by bash, not mangled
# by the Windows SSH quoting layer.
set -x
mkdir -p /mnt/vault/tmp/bc-runs /mnt/vault/tmp/bc-worktrees
nohup bash /home/audumla/_hw_rd13_gfx1030.sh > /mnt/vault/tmp/bc-runs/rd13-gfx1030-run1.log 2>&1 &
RD13_PID=$!
nohup bash /home/audumla/_hw_rd26_gfx1100.sh > /mnt/vault/tmp/bc-runs/rd26-gfx1100-run1.log 2>&1 &
RD26_PID=$!
echo "RD13_PID=$RD13_PID"
echo "RD26_PID=$RD26_PID"
sleep 2
echo "=== alive check ==="
for p in "$RD13_PID" "$RD26_PID"; do
  if kill -0 "$p" 2>/dev/null; then echo "PID $p ALIVE"; else echo "PID $p DEAD"; fi
done
