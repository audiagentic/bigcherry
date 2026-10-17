# RAD03 GPU mailbox

The Brutus feasibility probe in `mailbox_probe.cpp` measured 16.7 us round trip between an
RX 7900 XTX and the R9700 when both processes register the same POSIX shm page with
`hipHostRegisterPortable | hipHostRegisterMapped` and signal only with
`hipStreamWriteValue32` / `hipStreamWaitValue32`. P2P is unavailable on this host.

## Helper protocol v4

The helper registers the mailbox once and obtains its device pointer. Each producer rank has a
64-byte `LaneSignal` cache line and a private request-ring half. Each R9700 lane stream does:

1. wait for that rank's `request_sequence`;
2. read routes and BF16 activations directly from the mapped mailbox;
3. compact/sort routes on the GPU, run the resident-expert kernels, and apply the TP slice;
4. write the BF16 partial directly into the mapped result slot;
5. write `done_sequence = sequence`.

There is no activation H2D copy, result D2H copy, futex completion wait, or host synchronization in
that GPU batch. The helper host threads only turn structurally published descriptors into queued
batches and immediately advance to the next ring slot; they never wait for GPU completion. The
engine-side producer must publish descriptors early enough to keep at least two batches queued per
active lane and must not reuse a slot until its GPU-visible done sequence has passed it.

Expert load no longer maps the model container. The helper reads only metadata plus selected owned
expert planes with `pread`, through a 64 MiB pinned bounce allocation, relayouts one owned weight
at a time, uploads it asynchronously, and drops file cache after load. Startup logs
`host RSS after load`; the target is below 1 GiB.

## Pinned Radiance 1.3.0 primitive audit

Audited commit `89cee7ce414486c717aa2085c99e1f522e4c0ea6`.

What exists:

- `core/device/aql.cpp::barrier` emits an `HSA_PACKET_TYPE_BARRIER_AND` packet. It can wait on one
  `hsa_signal_t` and optionally complete another signal, with agent/system acquire/release scopes.
- `a_event_record` arms an engine-owned HSA signal and emits a barrier packet whose completion
  signal is that HSA signal.
- `a_event_wait` emits a barrier packet whose dependency is the latest event HSA signal.
- Public `abi/rad_device.h` exposes `rad_event_record` and `rad_event_wait`; the latter is a
  stream wait with no host synchronization.

What does not exist:

- no public or internal stream operation equivalent to `hipStreamWaitValue32` on an arbitrary
  mapped-host `uint32_t`;
- no stream operation equivalent to `hipStreamWriteValue32` to an arbitrary mapped-host
  `uint32_t`;
- no ABI for importing the other process's HSA signal into a `RadEvent`.

The existing event path therefore cannot represent the POSIX-shm sequence words used by the
cross-process mailbox. Calling HIP stream operations on the engine stream is not valid: Radiance
owns that stream as its AQL `Stream`/HSA queue.

**Current path:** patch rad0002 (radiance 1.7.0: thin hooks into `bc_moe_remote_*`) uses libr3 device ops on the rank stream. Publish masks helper-owned
route ids to -1 and re-scatters the remaining local ids. Local expert work then runs unchanged.
For a remote pass, the engine forces the ordinary MoE gather so the token output is materialized;
`moe_mailbox_add` waits for the lane completion and adds each helper result using the published
`Route::weight_bf16`. Contributions are accumulated in fp32 for each token in ascending top-k slot
order and the token row is written once before the standalone all-reduce and residual/HyperConn
write. HyperConn's fused gather+all-reduce and fused all-reduce write are disabled for that pass.

## Mandatory remote-result proof

Set `R3_MOE_HELPER_POISON=1` on the helper and engine environment. The helper replaces every
remote result element it writes with BF16 1000 and the engine hook prints exactly one
`libr3: R3_MOE_HELPER poison` marker at mailbox registration. Run the normal helper-on KL workload
against the helper-off reference. The poison run must diverge catastrophically (PPL far above the
reference); a non-poison helper run must return to the helper-off quality envelope. This proves the
R9700 result crosses the mailbox, is weighted by the published route weight, and reaches the model
token output rather than being dropped by gather bookkeeping.
