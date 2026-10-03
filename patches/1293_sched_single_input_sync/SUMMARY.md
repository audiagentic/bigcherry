# 1293_sched_single_input_sync

**Status:** untested
**Plan item:** QFN01/RNX01

## What it does

`ggml_backend_sched` synchronizes an event-less split backend once per split before copying user inputs,
instead of before every input.

## Why

With `-sm tensor` the split backend is the meta backend, which has no events, so every user input copy began
with a synchronize of every rank. A `hipStreamSynchronize` call-site trace of Flash-Next MTP decode
(`tools/lab/flash-next/sync-tracer.c`) showed ~485 host syncs per MTP step, almost all from this site and
the meta buffer `set_tensor`; the host spent ~82% of decode wall time inside the HIP API. One synchronize
already guarantees the previous use of every input buffer has finished.

Validation: decode t/s and ms/step A/B at 10K and 80K with MTP3; greedy parity without MTP at 10K; sync
count per step from the tracer.
