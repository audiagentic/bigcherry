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

## Result (2026-10-03, flashnext-sync-ab-1, ABBA on 1291+1292)

Neutral. ms per MTP step 10K: base 50.4/50.8 vs new 50.6/50.1; 80K: base 65.0/65.4 vs new 65.2/65.5.
hipStreamSynchronize per 256-token decode: 40325 -> 32894 (-18%). Greedy output without MTP identical at 10K.
The removed syncs were waits on GPU work the next step needs anyway, so host blocking dropped without
shortening the critical path. Not part of any deployment recipe; kept as a correct, upstreamable cleanup.
