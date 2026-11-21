# rad0055_step_phase_trace

**Status:** validated
**Plan item:** RUM09

Scope: Per-rank, per-op decode timing for a native serving start, selected by `--step-phase-trace`.

## Summary

The serving owner passes an authenticated run-directory path to the engine only when `--step-phase-trace` is given on a
native `serve` or `start`. The engine then brackets recorded device stretches, live passes and KV copy/wait intervals
with HIP events and writes one JSONL file per rank.

## Outcome

Without the switch the engine takes the upstream paths: no event is created, no file is opened, and the recorded-pass
replay is the upstream call. It needs no lab experiment, environment variable or YAML key, so it works with a native
configuration. The Radiance 2.2 warm decode gap is the question it exists to answer; it carries no throughput claim.

## Detail

The serving owner admits an explicit `--step-phase-trace` switch on
native v3 `serve` and `start`. The strict manifest reader accepts only
an authenticated owner-created `--bc-step-phase-dir` path; it is not
another configuration parser or lab experiment. The engine writes per-pass
JSONL files under `<run-dir>/step-phases/engine-rank-<n>.jsonl`.

When off, the engine's normal recorded-pass path does not allocate new HIP
events or create any log. When on, it uses an independent HIP event ring to bracket recorded device stretches,
live-pass envelopes, and explicit KV copy/wait intervals; it synchronises
at the end of each pass. **This perturbs timing, but preserves recorded-pass playback**; compare
cold versus warm within a traced build and use an untraced run for throughput.

`compute_ms` and `collective_ms` sum device-stretch intervals, including
parallel lanes; they are not a non-overlapping wall-time decomposition.
`idle_wait_ms` sums gaps between per-lane events, including host scheduling
and device waits. `host_kv_serial_ms` brackets on-stream KV copies;
`host_kv_wait_ms` brackets compute-stream waits for async DMA. Direct
asynchronous DMA activity cannot be timed with compute-stream HIP event
pairs; `host_kv_dma_ms` is always null, never falsely reported as zero.
A null value indicates no events in a class; `valid=false` marks overflow
or failed collection and must be excluded.

For replayed passes, tape_segments lists ordered HIP-event spans with immutable
packet boundaries (from, to) and elapsed ms; live passes have -1
boundaries. The packet bounds describe recording partitions, not individual
op labels or guaranteed nonoverlapping streams. They are enabled only with
the same diagnostic switch.

When tracing a replayed pass, its immutable recorded operation packet
ranges are replayed in their original order with per-op HIP event pairs
on the recorded issue lane. Each tape_segments record then carries the
source operation name. Gap packets are labeled unclassified, not
silently charged to compute or collectives. Timing is more perturbed
than coarse packet-range tracing but recording/replay remains active.
