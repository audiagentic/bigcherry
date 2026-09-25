#!/bin/bash
# Run a lane of campaign jobs one after another (one lane per GPU group).
# Usage: queue.sh <jobs-file>
# Each non-comment line: <patch> <producer|-> <arch> <device> <run-name> [extra campaign args...]
# A line whose run-name already has a finished log (CAMPAIGN_EXIT) is skipped,
# so a lane can be restarted after an interruption. Host paths come from the
# environment as for run_campaign.sh; BC_MODEL may be overridden per line with
# a leading MODEL=<path> token.
set -u
jobs=$1
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
while read -r line; do
    case "$line" in ''|'#'*) continue ;; esac
    set -- $line
    model=$BC_MODEL
    case "$1" in MODEL=*) model=${1#MODEL=}; shift ;; esac
    run=$5
    log="$root/work/runs/$run.log"
    if [ -f "$log" ] && grep -q '^CAMPAIGN_EXIT=' "$log"; then
        echo "skip $run (finished)"; continue
    fi
    echo "start $run $(date -Is)"
    BC_MODEL=$model bash "$here/run_campaign.sh" "$@" > "$log" 2>&1 < /dev/null
    # Attested llama-servers run at --verbosity 5 and log every full-vocab
    # response (~1.5 GB per arm); keep the head (startup, device attestation,
    # activation markers) and cap the rest so a lane cannot fill the disk.
    find "$root/work/runs/$run" -name '*server*.log' -size +20M -exec truncate -s 20M {} +
    echo "done  $run $(date -Is) $(tail -1 "$log")"
done < "$jobs"
