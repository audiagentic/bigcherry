# 1326_sched_async_host_inputs

**Status:** untested
**Plan item:** QFP16

## What it does

With `BIGCHERRY_SCHED_ASYNC_INPUTS=1`, ggml_backend_sched_compute_splits copies a split input whose source buffer is
host-resident (non-weights, contiguous) with ggml_backend_tensor_set_async on the split backend instead of
synchronizing the split backend and copying synchronously; the meta (-sm tensor) backend fans the copy out to every
device stream, and its set_tensor_async now falls back to the synchronous buffer path for split states it cannot
splice instead of aborting. 1325 measured ~2.6 ms per target verify round and ~0.4 ms per draft call in these copies.
