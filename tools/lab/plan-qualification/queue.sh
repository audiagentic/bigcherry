#!/bin/bash
# Run a mixed plan-qualification queue.
# Usage: queue.sh <jobs-file>
# Each non-comment line: <patch> <producer|-> <arch> <device> <run-name> [extra campaign args...]
# PROFILE lines use: PROFILE <patch> <arch> <device> <prefill|decode> <run-name> [args...]
# Optional leading MODEL=, HIP= and VIS= tokens are supported for both.
#
# Scheduling policy:
#   1. PROFILE build preparation fans out first. Same architecture+toolchain
#      build roots serialize behind profile_run.sh's exclusive build-key lock;
#      different build keys may prepare concurrently.
#   2. Successfully prepared PROFILE traces then fan out concurrently. They
#      hold shared build-key locks, a shared host-activity gate, and per-GPU
#      locks, so same-build traces can overlap on different cards without a
#      concurrent CMake/Ninja mutation.
#   3. after all profiles finish, campaign jobs run serially. run_campaign.sh
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

parse_prefixes() {
    local line=$1
    set -- $line
    PARSED_MODEL=$BC_MODEL
    PARSED_HIP=$BC_HIP_PATH
    PARSED_VIS=""
    while :; do
        case "$1" in
            MODEL=*) PARSED_MODEL=${1#MODEL=}; shift ;;
            HIP=*) PARSED_HIP=${1#HIP=}; shift ;;
            VIS=*) PARSED_VIS=${1#VIS=}; shift ;;
            *) break ;;
        esac
    done
    PARSED_ARGS=("$@")
}

profile_line() {
    local line=$1 phase=$2 run log rc
    parse_prefixes "$line"
    set -- "${PARSED_ARGS[@]}"
    [ "$1" = PROFILE ] || { echo "internal error: non-PROFILE row in profile phase" >&2; return 2; }
    shift
    run=$5
    log="$work/runs/$run.log"
    if [ -f "$log" ] && grep -q '^PROFILE_EXIT=' "$log"; then
        echo "skip profile $run (finished)"
        return 0
    fi
    if [ -n "$PARSED_VIS" ]; then export HIP_VISIBLE_DEVICES=$PARSED_VIS; else unset HIP_VISIBLE_DEVICES; fi
    if [ "$phase" = prepare ]; then
        echo "prepare profile $run $(date -Is)"
        BC_MODEL=$PARSED_MODEL BC_HIP_PATH=$PARSED_HIP BC_PROFILE_PHASE=prepare \
          bash "$here/profile_run.sh" "$@" > "$log" 2>&1 < /dev/null
        rc=$?
        echo "prepared profile $run $(date -Is) rc=$rc $(tail -1 "$log" 2>/dev/null || true)"
        return "$rc"
    fi
    echo "start profile $run $(date -Is)"
    BC_MODEL=$PARSED_MODEL BC_HIP_PATH=$PARSED_HIP BC_PROFILE_PHASE=run \
      bash "$here/profile_run.sh" "$@" >> "$log" 2>&1 < /dev/null
    rc=$?
    echo "done  profile $run $(date -Is) rc=$rc $(tail -1 "$log" 2>/dev/null || true)"
    return "$rc"
}

campaign_line() {
    local line=$1 run log rc
    parse_prefixes "$line"
    set -- "${PARSED_ARGS[@]}"
    run=$5
    log="$work/runs/$run.log"
    if [ -f "$log" ] && grep -q '^CAMPAIGN_EXIT=' "$log"; then
        echo "skip campaign $run (finished)"
        return 0
    fi
    echo "start campaign $run $(date -Is)"
    if [ -n "$PARSED_VIS" ]; then export HIP_VISIBLE_DEVICES=$PARSED_VIS; else unset HIP_VISIBLE_DEVICES; fi
    BC_MODEL=$PARSED_MODEL BC_HIP_PATH=$PARSED_HIP bash "$here/run_campaign.sh" "$@" > "$log" 2>&1 < /dev/null
    rc=$?
    # Attested llama-servers can emit enormous full-vocab logs. Preserve the
    # startup/device/activation head and cap the remainder after each run.
    find "$work/runs/$run" -name '*server*.log' -size +20M -exec truncate -s 20M {} + 2>/dev/null || true
    echo "done  campaign $run $(date -Is) rc=$rc $(tail -1 "$log" 2>/dev/null || true)"
    return "$rc"
}

failures=0
ready_profiles=()
if ((${#profiles[@]})); then
    echo "profile prepare phase: ${#profiles[@]} job(s)"
    pids=()
    for line in "${profiles[@]}"; do
        profile_line "$line" prepare &
        pids+=("$!")
    done
    for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then
            ready_profiles+=("${profiles[$i]}")
        else
            failures=$((failures + 1))
        fi
    done
fi

if ((${#ready_profiles[@]})); then
    echo "profile trace phase: ${#ready_profiles[@]} prepared job(s), parallel across GPU locks"
    pids=()
    for line in "${ready_profiles[@]}"; do
        profile_line "$line" run &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then failures=$((failures + 1)); fi
    done
fi

if ((${#campaigns[@]})); then
    echo "campaign phase: ${#campaigns[@]} host-isolated job(s)"
    for line in "${campaigns[@]}"; do
        if ! campaign_line "$line"; then failures=$((failures + 1)); fi
    done
fi

if ((failures)); then
    echo "queue completed with $failures failed job(s)"
    exit 1
fi
echo "queue completed successfully"
