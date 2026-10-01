# Profiling lab

`profile-27b-q8.sh` — rocprofv3 runtime trace + stats for a 64-token decode and one
2048-token prefill ubatch of Qwen3.8-27B Q8_0 on dual gfx1100 (`-sm tensor`), plus an SQ counter
run. Queued as a SCRIPT row (`queue-27b-profile.sh`) so it holds the host and GPU locks.

Attribution method: for decode, take the repeating 64-token region and group per-device
critical-path kernel time into `mul_mat_vec_q*`, `gated_delta_net*`, AllReduce kernels/copies,
`flash_attn*`, and other; per device, not summed across GPUs; launch gap = idle time between
dependent dispatches on the critical queue; divide by 64 for per-token cost.
