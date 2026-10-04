# 1326_sched_async_host_inputs

**Status:** evaluated
**Plan item:** QFP16

## What it does

With `BIGCHERRY_SCHED_ASYNC_INPUTS=1`, ggml_backend_sched_compute_splits copies a split input whose source buffer is
host-resident (non-weights, contiguous) with ggml_backend_tensor_set_async on the split backend instead of
synchronizing the split backend and copying synchronously; the meta (-sm tensor) backend fans the copy out to every
device stream, and its set_tensor_async now falls back to the synchronous buffer path for split states it cannot
splice instead of aborting. 1325 measured ~2.6 ms per target verify round and ~0.4 ms per draft call in these copies.

## Hardware result (2026-10-04, flashnext-1326-d24k/d80k, env screen on one v4+1326 build)

~24K 44.3/44.5 -> 40.0 ms/step (-10%), 72.2/71.9 -> 78.0 t/s; ~80K 50.6/50.0 -> 46.7 ms/step (-7%), 58.8/60.2 -> 64.5 t/s; greedy identical at both depths; acceptance equal. Target submit 5.7 -> 3.0 ms/round; meta split input handling 2.56 -> 0.38 ms; draft split input 0.45-0.57 -> 0.014-0.016 ms/call. Profile v5 candidate (with the prefill patches 1237/1265/1253 if their screen is clean).
