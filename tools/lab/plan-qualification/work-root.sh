#!/bin/bash
# Shared support library for the transitional plan-qualification queue.
# Executed directly, prints the campaign work root. Sourced by queue launchers,
# it also provides host-activity and per-GPU flock primitives.

work_root_resolve() {
    local root=$1
    if [ -n "${BIGCHERRY_WORK_ROOT:-}" ]; then
        printf '%s\n' "$BIGCHERRY_WORK_ROOT"
        return 0
    fi
    python3 - "$root" <<'PY'
import sys, tomllib
from pathlib import Path
root = Path(sys.argv[1])
local = root / "config" / "environment.local.toml"
env = tomllib.loads(local.read_text())["env"] if local.is_file() else {}
print(env.get("BIGCHERRY_WORK_ROOT") or root / "work")
PY
}

# Per-device exclusive locks. Multi-device jobs acquire numeric device ids in a
# deterministic order so overlapping requests cannot deadlock.
_GPU_LOCK_FDS=()

gpu_lock_acquire() {
    local work=$1 devices=$2 dir dev fd
    dir="$work/queue/gpu-locks"
    mkdir -p "$dir"
    for dev in $(printf '%s\n' "${devices//,/$'\n'}" | sort -n); do
        exec {fd}>"$dir/dev-$dev.lock"
        flock "$fd"
        _GPU_LOCK_FDS+=("$fd")
    done
}

gpu_lock_release() {
    local fd
    for fd in "${_GPU_LOCK_FDS[@]:-}"; do
        [ -n "$fd" ] && eval "exec $fd>&-"
    done
    _GPU_LOCK_FDS=()
}

# Host-wide reader/writer activity gate. PROFILE preparation/traces are readers;
# monolithic validation campaigns are writers. Writer intent prevents an
# arriving stream of readers from starving a pending timed campaign.
_ACTIVITY_LOCK_FDS=()

_activity_close_fd() {
    local fd=$1
    [ -n "$fd" ] && eval "exec $fd>&-"
}

_activity_paths() {
    local work=$1
    _ACTIVITY_DIR="$work/queue/activity"
    _ACTIVITY_GATE="$_ACTIVITY_DIR/gate.lock"
    _ACTIVITY_WRITER="$_ACTIVITY_DIR/writer.lock"
    _ACTIVITY_INTENT="$_ACTIVITY_DIR/performance.intent"
    mkdir -p "$_ACTIVITY_DIR"
}

_activity_repair_stale_intent() {
    local probe
    [ -e "$_ACTIVITY_INTENT" ] || return 0
    exec {probe}>"$_ACTIVITY_WRITER"
    if flock -n "$probe"; then
        rm -f "$_ACTIVITY_INTENT"
        _activity_close_fd "$probe"
        return 0
    fi
    _activity_close_fd "$probe"
    return 1
}

activity_lock_shared_acquire() {
    local work=$1 fd
    _activity_paths "$work"
    while :; do
        if [ -e "$_ACTIVITY_INTENT" ]; then
            _activity_repair_stale_intent || { sleep 0.05; continue; }
        fi
        exec {fd}>"$_ACTIVITY_GATE"
        flock -s "$fd"
        # Close the check->lock race: if a writer published intent while this
        # reader acquired the gate, step aside so the writer can drain readers.
        if [ ! -e "$_ACTIVITY_INTENT" ]; then
            _ACTIVITY_LOCK_FDS+=("$fd")
            return 0
        fi
        _activity_close_fd "$fd"
        sleep 0.05
    done
}

activity_lock_exclusive_acquire() {
    local work=$1 writer_fd gate_fd tmp
    _activity_paths "$work"
    exec {writer_fd}>"$_ACTIVITY_WRITER"
    flock "$writer_fd"
    tmp="$_ACTIVITY_INTENT.$$"
    printf 'pid=%s\nstarted=%s\n' "$$" "$(date -Is)" > "$tmp"
    mv -f "$tmp" "$_ACTIVITY_INTENT"
    exec {gate_fd}>"$_ACTIVITY_GATE"
    flock "$gate_fd"
    rm -f "$_ACTIVITY_INTENT"
    _ACTIVITY_LOCK_FDS+=("$gate_fd" "$writer_fd")
}

activity_lock_release() {
    local fd
    for fd in "${_ACTIVITY_LOCK_FDS[@]:-}"; do
        [ -n "$fd" ] && _activity_close_fd "$fd"
    done
    _ACTIVITY_LOCK_FDS=()
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    if [ "$#" -ne 1 ]; then
        echo "usage: work-root.sh <repo-root>" >&2
        exit 2
    fi
    work_root_resolve "$1"
fi
