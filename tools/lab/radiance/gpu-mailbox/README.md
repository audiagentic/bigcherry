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

**Result:** patch 0002 is not changed to the GPU mailbox path in this slice. The primitive required
by the owner direction is absent. Protocol-v4 libr3 deliberately refuses to register the legacy
CPU-mediated D2H/futex/H2D hook rather than silently retaining it.

## Patch 0002 change once the AQL primitive exists

Add two Radiance device operations with system-scope ordering on mapped host memory, implemented by
the AQL backend rather than forwarded to HIP:

- stream wait: do not pass until `*addr >= value`;
- stream write: store `value` to `*addr` after prior queue writes are system-visible.

Then replace `core/place/moe_remote.cpp`'s current device-to-host copies, `rad_stream_sync`, CPU
route loop, hook submit, CPU wait, and host-to-device route-mask copy with this stream-ordered path:

1. issue an on-GPU route-select/compact kernel on the rank stream. It reads `ids_dev` and
   `ew_dev`, applies the helper ownership bitmap, masks selected local IDs in place, and writes
   compact `Route` records to that rank's mapped mailbox slot;
2. copy the activation to the mapped mailbox with the engine GPU stream (GPU to host-mapped memory;
   no CPU copy), then stream-write `request_sequence`;
3. continue queuing local work while the R9700 lane stream waits, computes and stream-writes
   `done_sequence`;
4. immediately before the remote partial is merged, stream-wait on `done_sequence`, copy/read the
   mapped result into the rank's remote-partial buffer, then issue the existing add and all-reduce.

The sequence words need system-scope release/acquire semantics because both GPUs access pinned host
memory and P2P is unavailable. Route selection and mailbox readiness must remain on the GPU; a host
callback or host event wait would reintroduce the service latency RAD03 removes.
