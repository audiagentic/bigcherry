#!/bin/bash
# Run a command once earlier queue runs have finished, without matching process names.
#
# Waiting on `pgrep -f <pattern>` is how chains deadlock: the waiter's own command line, or another waiter's,
# contains the pattern (2026-10-11: a chain and a stale waiter held each other for 43 minutes; twice a
# `pkill -f` ended the ssh session that issued it). This waits on what the runs themselves write instead: a queue
# log is finished when it has a QUEUE_EXIT= or ALL_JOBS_DONE line or the queue's "queue completed" line.
#
# Usage: after.sh <log>... -- <command> [args...]
#   after.sh runs/a-queue.log runs/b-queue.log -- bash tools/lab/flash-next/queue-env-ab.sh c b-x 24576
# env: AFTER_POLL (seconds between checks, 30), AFTER_MAX (give up after this many seconds, 0 = never)
set -u
logs=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do logs+=("$1"); shift; done
[ "${1:-}" = "--" ] || { echo "usage: after.sh <log>... -- <command> [args...]"; exit 2; }
shift
[ $# -gt 0 ] || { echo "after.sh: no command"; exit 2; }
t0=$(date +%s)
for log in "${logs[@]}"; do
    until grep -qE "^QUEUE_EXIT=|^ALL_JOBS_DONE|^queue completed" "$log" 2>/dev/null; do
        if [ "${AFTER_MAX:-0}" -gt 0 ] && [ $(( $(date +%s) - t0 )) -ge "${AFTER_MAX}" ]; then
            echo "after.sh: gave up waiting for $log"; exit 3
        fi
        sleep "${AFTER_POLL:-30}"
    done
done
exec "$@"
