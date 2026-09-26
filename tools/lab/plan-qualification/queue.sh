#!/bin/bash
# Run a lane of campaign jobs one after another (one lane per GPU group).
# Usage: queue.sh <jobs-file>
# Each non-comment line: <patch> <producer|-> <arch> <device> <run-name> [extra campaign args...]
# A line whose run-name already has a finished log (CAMPAIGN_EXIT) is skipped,
# so a lane can be restarted after an interruption. Host paths come from the
# environment as for run_campaign.sh; BC_MODEL may be overridden per line with
# a leading MODEL=<path> token and the ROCm prefix with HIP=<path>.
set -u
jobs=$1
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
work=$("$root/tools/lab/plan-qualification/work-root.sh" "$root")
mkdir -p "$work/runs"
while read -r line; do
    case "$line" in ''|'#'*) continue ;; esac
    set -- $line
    model=$BC_MODEL
    hip=$BC_HIP_PATH
    vis=""
    # Optional leading tokens, in any order: MODEL=<gguf> HIP=<rocm prefix>
    # VIS=<HIP_VISIBLE_DEVICES> (multi-GPU jobs)
    while :; do
        case "$1" in
            MODEL=*) model=${1#MODEL=}; shift ;;
            HIP=*) hip=${1#HIP=}; shift ;;
            VIS=*) vis=${1#VIS=}; shift ;;
            *) break ;;
        esac
    done
    # PROFILE <patch-id> <arch> <device> <prefill|decode> <run-name> [args...]:
    # PVPS10 kernel profile of control vs subject (profile_run.sh), run between
    # jobs so it never overlaps a measurement.
    kind=campaign
    if [ "$1" = PROFILE ]; then kind=profile; shift; fi
    run=$5
    log="$work/runs/$run.log"
    if [ -f "$log" ] && grep -q '^\(CAMPAIGN\|PROFILE\)_EXIT=' "$log"; then
        echo "skip $run (finished)"; continue
    fi
    echo "start $run $(date -Is)"
    if [ -n "$vis" ]; then export HIP_VISIBLE_DEVICES=$vis; else unset HIP_VISIBLE_DEVICES; fi
    if [ "$kind" = profile ]; then
        BC_MODEL=$model BC_HIP_PATH=$hip bash "$here/profile_run.sh" "$@" > "$log" 2>&1 < /dev/null
    else
        BC_MODEL=$model BC_HIP_PATH=$hip bash "$here/run_campaign.sh" "$@" > "$log" 2>&1 < /dev/null
    fi
    # Attested llama-servers run at --verbosity 5 and log every full-vocab
    # response (~1.5 GB per arm); keep the head (startup, device attestation,
    # activation markers) and cap the rest so a lane cannot fill the disk.
    find "$work/runs/$run" -name '*server*.log' -size +20M -exec truncate -s 20M {} +
    echo "done  $run $(date -Is) $(tail -1 "$log")"
done < "$jobs"
