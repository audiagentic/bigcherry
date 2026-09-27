#!/bin/bash
# Run a mixed plan-qualification queue.
# Usage: queue.sh <jobs-file>
# Each non-comment line: <patch> <producer|-> <arch> <device> <run-name> [extra campaign args...]
# PROFILE lines use: PROFILE <patch> <arch> <device> <prefill|decode> <run-name> [args...]
# Optional leading MODEL=, HIP= and VIS= tokens are supported for both.
#
# Scheduling policy:
#   1. all PROFILE diagnostics are launched concurrently; per-GPU locks cap
#      actual GPU concurrency and allow different cards to profile in parallel;
#   2. after profiles finish, campaign jobs run serially. run_campaign.sh also
#      takes the host-exclusive activity gate, so multiple queue.sh processes
#      cannot accidentally overlap timed campaigns.
# Finished run logs are skipped, so the queue is restartable. Independent jobs
# continue after a failure, but the queue itself exits nonzero when any launched
# job failed so humans/agents/systemd can detect an unhealthy batch.
set -u
jobs=$1
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
work=$("$root/tools/lab/plan-qualification/work-root.sh" "$root")
mkdir -p "$work/runs"

profiles=()
campaigns=()
while IFS= read -r line; do
    case "$line" in ''|'#'*) continue ;; esac
    set -- $line
    while :; do
        case "$1" in MODEL=*|HIP=*|VIS=*) shift ;; *) break ;; esac
    done
    if [ "$1" = PROFILE ]; then profiles+=("$line"); else campaigns+=("$line"); fi
done < "$jobs"

run_line() {
    local line=$1 model hip vis kind run log rc
    set -- $line
    model=$BC_MODEL
    hip=$BC_HIP_PATH
    vis=""
    while :; do
        case "$1" in
            MODEL=*) model=${1#MODEL=}; shift ;;
            HIP=*) hip=${1#HIP=}; shift ;;
            VIS=*) vis=${1#VIS=}; shift ;;
            *) break ;;
        esac
    done
    kind=campaign
    if [ "$1" = PROFILE ]; then kind=profile; shift; fi
    run=$5
    log="$work/runs/$run.log"
    if [ -f "$log" ] && grep -q '^\(CAMPAIGN\|PROFILE\)_EXIT=' "$log"; then
        echo "skip $run (finished)"
        return 0
    fi
    echo "start $kind $run $(date -Is)"
    if [ -n "$vis" ]; then export HIP_VISIBLE_DEVICES=$vis; else unset HIP_VISIBLE_DEVICES; fi
    if [ "$kind" = profile ]; then
        BC_MODEL=$model BC_HIP_PATH=$hip bash "$here/profile_run.sh" "$@" > "$log" 2>&1 < /dev/null
        rc=$?
    else
        BC_MODEL=$model BC_HIP_PATH=$hip bash "$here/run_campaign.sh" "$@" > "$log" 2>&1 < /dev/null
        rc=$?
    fi
    # Attested llama-servers can emit enormous full-vocab logs. Preserve the
    # startup/device/activation head and cap the remainder after each run.
    find "$work/runs/$run" -name '*server*.log' -size +20M -exec truncate -s 20M {} + 2>/dev/null || true
    echo "done  $kind $run $(date -Is) rc=$rc $(tail -1 "$log" 2>/dev/null || true)"
    return "$rc"
}

failures=0
if ((${#profiles[@]})); then
    echo "profile phase: ${#profiles[@]} job(s), parallel across GPU locks"
    pids=()
    for line in "${profiles[@]}"; do
        run_line "$line" &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then failures=$((failures + 1)); fi
    done
fi

if ((${#campaigns[@]})); then
    echo "campaign phase: ${#campaigns[@]} host-isolated job(s)"
    for line in "${campaigns[@]}"; do
        if ! run_line "$line"; then failures=$((failures + 1)); fi
    done
fi

if ((failures)); then
    echo "queue completed with $failures failed job(s)"
    exit 1
fi
echo "queue completed successfully"
