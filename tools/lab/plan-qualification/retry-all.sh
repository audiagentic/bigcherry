#!/bin/bash
# Wait for every running queue to finish, clear failed rows, and re-run the listed job scripts
# in order (successful rows are skipped). Optionally run a final command (e.g. restart a
# service that was stopped for the tests).
# Usage: retry-all.sh "<glob> ..." "<job-script> ..." [final command...]
set -u
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
cd "$root"
globs=$1 scripts=$2
shift 2
while ps -eo cmd | grep -q "[p]lan-qualification/queue.sh"; do sleep 60; done
bash "$here/retry-failed.sh" $globs
for s in $scripts; do
    echo "== $s $(date -Is)"
    bash "$s"
done
if [ $# -gt 0 ]; then "$@"; fi
echo RETRY_ALL_DONE
