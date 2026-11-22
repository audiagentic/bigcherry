# Placement worker idle cost

Question: the engine's placement thread (`placement_loop`, `moe_remote_hook.cpp`) snapshots the placement control page,
steps the coordinator and sleeps 2 ms, including when no placement cycle is running. Is that idle poll worth replacing
with a wake from the generation notify futex. Host-only, CPU-only, no GPU.

## Measurement

`idle_bench.cpp` (this folder) runs the idle loop body for 30 s: a seqlock snapshot of one 128-byte control page, one
mutex lock and unlock (the ledger's `stopped()` check), then a 2 ms sleep. It reports the thread's own CPU time.

| Quantity | Value |
|---|---|
| Wakeups | 14,581 in 30 s (486 per second) |
| Thread CPU | 83.9 ms in 30 s |
| Share of one core | 0.28 % per engine process |

This is the loop body only. It is not an idle engine measurement; the real step adds one more ledger lock.

Latency: a placement cycle has about six transitions (RESERVING, COPYING, QUIESCING, COMMIT, RETIRING, ACTIVE), each
seen up to 2 ms late. Placement runs off the serving path, so this is not decode latency.

## Decision

Not worth changing. 0.28 % of one core is negligible, and a wake-based idle path would need a new
idle-versus-polling signal from the coordinator, a notify in `stop_workers` (the placement thread is joined before the
retire wake), and care over lost wakes.

## The ack loop in `generation_close.h` stays a poll

`await_drain` polls the helper acknowledgement every 2 ms during a transition. Waking it from the header notify is not
possible without a helper protocol change:

- the helper's drain acknowledgement is a plain release store with no notify (`acknowledge_lane_drain`,
  `moe_remote_shm.h`);
- `helper_drained` (`moe_remote_hook.cpp`) also reads the GPU-written `consumed_sequence` of each lane
  (`lane_quiesced`, `moe_remote_shm.h`), which completes with no CPU notification at all.

A GPU event that completes without a CPU notification needs a bounded poll; this is that case.
