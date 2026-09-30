#!/bin/bash
# Clear failed queue rows so a re-run of their job script retries them.
# Usage: retry-failed.sh <run-name-glob>...   e.g. retry-failed.sh 'b-27b-*' 'pf-27b-*' 'ab-27b-*'
# A row counts as failed when its log ends a *_EXIT= line with a non-zero code (or never wrote one
# because it was blocked). Successful rows are left untouched, so their results are reused.
set -u
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
source "$here/work-root.sh"
runs="$(work_root_resolve "$root")/runs"
for glob in "$@"; do
    for log in "$runs"/$glob.log; do
        [ -f "$log" ] || continue
        if grep -qE '^(BUILD|AB|PREFLIGHT|CAMPAIGN|PROFILE)_EXIT=0$' "$log"; then continue; fi
        run=$(basename "$log" .log)
        echo "clearing failed row $run"
        rm -rf "$log" "$runs/$run"
    done
done
