#!/bin/bash
# Run a mixed plan-qualification queue.
# Usage: queue.sh <jobs-file>
# Each non-comment line: <patch> <producer|-> <arch> <device> <run-name> [extra campaign args...]
# PROFILE lines use: PROFILE <patch> <arch> <device> <prefill|decode> <run-name> [args...]
# Optional leading MODEL=, HIP= and VIS= tokens are supported for both.
# BUILD <run-name> <experiment|-> [arch-list] [binary] builds bigcherry:stock:linux-multi
#   (+ <experiment>) for arch-list (comma-separated, default gfx1100; e.g. gfx1100,gfx1201 for
#   XTX+R9700), target binary default bin/llama-server (e.g. bin/llama-perplexity), and records
#   BUILD_BINARY=<path>.
# SCRIPT <run-name> <script> [args...] runs a repo script under the host + GPU locks; @<build-run>
#   arguments are resolved to binaries. REQUIRES= gates it like AB rows.
# AB <run-name> <server-config.json> [ab-benchmark args] runs a balanced server A/B.
# VIS=<gpus> on BUILD/AB/PREFLIGHT rows selects the GPU set they lock and (preflight) run on;
#   default 0,1. AB topology itself comes from the config's environment block.
# PREFLIGHT and AB binary arguments (and "@name" strings inside an AB config) may be
#   @<build-run-name>, resolved to that BUILD row's BUILD_BINARY.
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
source "$here/work-root.sh"
work=$(work_root_resolve "$root")
mkdir -p "$work/runs"

profiles=()
campaigns=()
while IFS= read -r line; do
    case "$line" in ''|'#'*) continue ;; esac
    set -- $line
    while :; do
        case "$1" in MODEL=*|HIP=*|VIS=*|REQUIRES=*) shift ;; *) break ;; esac
    done
    if [ "$1" = PROFILE ]; then profiles+=("$line"); else campaigns+=("$line"); fi
done < "$jobs"

parse_prefixes() {
    local line=$1
    set -- $line
    PARSED_MODEL=$BC_MODEL
    PARSED_HIP=$BC_HIP_PATH
    PARSED_VIS=""
    PARSED_REQUIRES=""
    while :; do
        case "$1" in
            MODEL=*) PARSED_MODEL=${1#MODEL=}; shift ;;
            HIP=*) PARSED_HIP=${1#HIP=}; shift ;;
            VIS=*) PARSED_VIS=${1#VIS=}; shift ;;
            REQUIRES=*) PARSED_REQUIRES=${1#REQUIRES=}; shift ;;
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

resolve_binary() {
    # @<build-run> -> BUILD_BINARY recorded by that BUILD row; anything else unchanged.
    case "$1" in
        @*) sed -n 's/^BUILD_BINARY=//p' "$work/runs/${1#@}.log" 2>/dev/null | tail -1 ;;
        *) echo "$1" ;;
    esac
}

build_line() {
    # BUILD <run-name> <experiment|-> [arch-list] [binary]   ("-" builds the plain lane with no experiment)
    local run=$2 arch=${4:-gfx1100} target=${5:-bin/llama-server} log rc plan bin
    local experiment_args=()
    [ "$3" != - ] && experiment_args=(--experiment "$3")
    log="$work/runs/$run.log"
    if [ -f "$log" ] && grep -q '^BUILD_EXIT=' "$log"; then
        echo "skip build $run (finished)"
        grep -qx 'BUILD_EXIT=0' "$log"
        return $?
    fi
    echo "start build $run $(date -Is)"
    ROCM_PATH=/opt/rocm PYTHONPATH="$root/tools" bash "$here/locked-run.sh" \
        python3 -m bigcherry build --lane bigcherry:stock:linux-multi "${experiment_args[@]}" \
        --arch "$arch" --binary-relative-path "$target" > "$log" 2>&1 < /dev/null
    rc=$?
    plan=$(sed -n 's/.*: ok build_plan_id=\([0-9a-f]*\).*/\1/p' "$log" | tail -1)
    bin=""
    # bigcherry build publishes under the checkout's work/builds (not the queue work root).
    local matches=()
    [ -n "$plan" ] && mapfile -t matches < <(ls -d "$root"/work/builds/*/"$plan"/"$target" 2>/dev/null)
    [ "${#matches[@]}" -eq 1 ] && bin=${matches[0]}
    if [ "$rc" -eq 0 ] && [ -f "$bin" ]; then echo "BUILD_BINARY=$bin" >> "$log"; else [ "$rc" -eq 0 ] && rc=1; fi
    echo "BUILD_EXIT=$rc" >> "$log"
    echo "done  build $run $(date -Is) rc=$rc $bin"
    return "$rc"
}

ab_line() {
    # AB <run-name> <server-config.json> [ab-benchmark args]
    local run=$2 config=$3 log rc out resolved ref bin
    shift 3
    log="$work/runs/$run.log"
    out="$work/runs/$run"
    if [ -f "$log" ] && grep -q '^AB_EXIT=' "$log"; then
        echo "skip ab $run (finished)"
        grep -qx 'AB_EXIT=0' "$log"
        return $?
    fi
    echo "start ab $run $(date -Is)"
    mkdir -p "$out"
    resolved="$out/server-config.json"
    cp "$config" "$resolved"
    for ref in $(grep -o '"@[A-Za-z0-9._-]*"' "$config" | tr -d '"' | sort -u); do
        bin=$(resolve_binary "$ref")
        if [ -z "$bin" ]; then
            echo "blocked ab $run: $ref has no BUILD_BINARY" | tee "$log"
            echo "AB_EXIT=1" >> "$log"
            return 1
        fi
        sed -i "s|\"$ref\"|\"$bin\"|g" "$resolved"
    done
    # Lock exactly the GPUs the config runs on; a VIS= prefix must agree with it.
    local cfg_gpus
    # Physical GPUs: ROCR_VISIBLE_DEVICES when set (HIP indices are renumbered after it).
    cfg_gpus=$(python3 -c 'import json,sys; e=json.load(open(sys.argv[1]))["environment"]; print(e.get("ROCR_VISIBLE_DEVICES") or e["HIP_VISIBLE_DEVICES"])' "$resolved")
    if [ -n "$PARSED_VIS" ] && [ "$PARSED_VIS" != "$cfg_gpus" ]; then
        echo "blocked ab $run: VIS=$PARSED_VIS disagrees with config HIP_VISIBLE_DEVICES=$cfg_gpus" | tee "$log"
        echo "AB_EXIT=1" >> "$log"
        return 1
    fi
    export BC_GPUS=$cfg_gpus
    ROCM_PATH=/opt/rocm PYTHONPATH="$root/tools" bash "$here/locked-run.sh" \
        python3 -m bigcherry ab-benchmark --server-config "$resolved" --output "$out/result" "$@" > "$log" 2>&1 < /dev/null
    rc=$?
    echo "AB_EXIT=$rc" >> "$log"
    echo "done  ab $run $(date -Is) rc=$rc"
    return "$rc"
}

script_line() {
    # SCRIPT <run-name> <script> [args...]
    local run=$2 script=$3 log rc arg bin
    shift 3
    log="$work/runs/$run.log"
    if [ -f "$log" ] && grep -q '^SCRIPT_EXIT=' "$log"; then
        echo "skip script $run (finished)"
        grep -qx 'SCRIPT_EXIT=0' "$log"
        return $?
    fi
    echo "start script $run $(date -Is)"
    local args=()
    for arg in "$@"; do
        case "$arg" in
            @*) bin=$(resolve_binary "$arg")
                if [ -z "$bin" ]; then
                    echo "blocked script $run: $arg has no BUILD_BINARY" | tee "$log"
                    echo "SCRIPT_EXIT=1" >> "$log"
                    return 1
                fi
                args+=("$bin") ;;
            *) args+=("$arg") ;;
        esac
    done
    mkdir -p "$work/runs/$run"
    BC_RUN_DIR="$work/runs/$run" bash "$here/locked-run.sh" bash "$script" "${args[@]}" > "$log" 2>&1 < /dev/null
    rc=$?
    echo "SCRIPT_EXIT=$rc" >> "$log"
    echo "done  script $run $(date -Is) rc=$rc"
    return "$rc"
}

preflight_line() {
    # PREFLIGHT <run-name> <binary> <model> <marker-regex> [server args...]
    # Proves the patch marker fires on the target model; campaign rows that
    # carry REQUIRES=<run-name> are refused unless this exited 0.
    local run=$2 log rc
    log="$work/runs/$run.log"
    if [ -f "$log" ] && grep -q '^PREFLIGHT_EXIT=' "$log"; then
        echo "skip preflight $run (finished)"
        grep -qx 'PREFLIGHT_EXIT=0' "$log"
        return $?
    fi
    echo "start preflight $run $(date -Is)"
    shift 2
    local bin
    bin=$(resolve_binary "$1")
    shift
    if [ -z "$bin" ]; then
        echo "blocked preflight $run: binary reference has no BUILD_BINARY" | tee "$log"
        echo "PREFLIGHT_EXIT=1" >> "$log"
        return 1
    fi
    bash "$here/../native-vs-patched/preflight-fire.sh" "$bin" "$@" > "$log" 2>&1 < /dev/null
    rc=$?
    echo "PREFLIGHT_EXIT=$rc" >> "$log"
    echo "done  preflight $run $(date -Is) rc=$rc"
    return "$rc"
}

campaign_line() {
    local line=$1 run log rc
    parse_prefixes "$line"
    set -- "${PARSED_ARGS[@]}"
    export BC_GPUS=${PARSED_VIS:-0,1}
    if [ -n "$PARSED_REQUIRES" ] && { [ "$1" = BUILD ] || [ "$1" = PREFLIGHT ]; }; then
        echo "invalid row $1 ${2:-}: REQUIRES= applies only to AB and campaign rows" >&2
        return 1
    fi
    if [ "$1" = BUILD ]; then build_line "$@"; return $?; fi
    if [ "$1" = PREFLIGHT ]; then preflight_line "$@"; return $?; fi
    if [ -n "$PARSED_REQUIRES" ] && ! grep -qx 'PREFLIGHT_EXIT=0' "$work/runs/$PARSED_REQUIRES.log" 2>/dev/null; then
        echo "blocked $1 ${2:-}: preflight $PARSED_REQUIRES did not pass (patch not proven to fire)"
        return 1
    fi
    if [ "$1" = AB ]; then ab_line "$@"; return $?; fi
    if [ "$1" = SCRIPT ]; then script_line "$@"; return $?; fi
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
