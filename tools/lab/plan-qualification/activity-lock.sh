#!/bin/bash
# Host-wide activity gate for the legacy plan-qualification queue.
#
# PROFILE/trace-style diagnostic jobs acquire SHARED mode: many may run at
# once on different GPUs (per-GPU locking is a separate layer).
# Performance campaigns acquire EXCLUSIVE mode: no profile/diagnostic holder
# may overlap, and only one campaign owns the host at a time.
#
# Writer intent is published before waiting for the exclusive gate so a
# continuous stream of new shared readers cannot starve a pending performance
# campaign. A stale intent left by a crashed writer is repaired when no live
# writer owns writer.lock.
#
# Source this file; descriptors remain owned by the caller until release/exit.
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
        # Close the check->lock race: if a writer published intent while the
        # shared lock was being acquired, step aside and let it drain readers.
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
