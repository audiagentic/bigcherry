#!/bin/bash
# Per-GPU-device exclusive locking, shared by run_campaign.sh (the isolated
# performance lane) and profile_run.sh (the parallel profile lane) so the two
# lanes can run concurrently without ever colliding on the same physical card
# -- a profile job and a timed perf session sharing a GPU would not just add
# noise, they would OOM/crash each other.
#
# Meant to be SOURCED, not executed, so the flock file descriptors it opens
# stay held for the rest of the caller's process (released automatically on
# exit, or explicitly with gpu_lock_release).
#
#   source .../gpu-lock.sh
#   gpu_lock_acquire "$work" "$devices"   # e.g. devices="0,1"; blocks until free
#   ... touch the GPU(s) ...
#   gpu_lock_release                      # optional; process exit does this too
#
# Devices are locked in ascending numeric order (never the job's own order),
# which is what makes a dual-device job and another job's single-device lock
# request converge on the same acquisition order and never deadlock.
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
