# 1326_sched_async_host_inputs

**Status:** validated
**Plan item:** QFP16

## What it does

With `BIGCHERRY_SCHED_ASYNC_INPUTS=1`, ggml_backend_sched_compute_splits copies a split input whose source buffer is
host-resident (non-weights, contiguous) with ggml_backend_tensor_set_async on the split backend instead of
synchronizing the split backend and copying synchronously; the meta (-sm tensor) backend fans the copy out to every
device stream, and its set_tensor_async now falls back to the synchronous buffer path for split states it cannot
splice instead of aborting. 1325 measured ~2.6 ms per target verify round and ~0.4 ms per draft call in these copies.

## Hardware result (2026-10-04, flashnext-1326-d24k/d80k, env screen on one v4+1326 build)

~24K 44.3/44.5 -> 40.0 ms/step (-10%), 72.2/71.9 -> 78.0 t/s; ~80K 50.6/50.0 -> 46.7 ms/step (-7%), 58.8/60.2 -> 64.5 t/s; greedy identical at both depths; acceptance equal. Target submit 5.7 -> 3.0 ms/round; meta split input handling 2.56 -> 0.38 ms; draft split input 0.45-0.57 -> 0.014-0.016 ms/call. Profile v5 candidate (with the prefill patches 1237/1265/1253 if their screen is clean).

Adopted in production profile v5 (2026-10-04, flashnext-v5-abba): v4 -> v5 decode +10.8% at ~8K, +7.8% at ~64K; prefill +3.7% / +7.2%; complete separation; greedy identical across 8 arms per depth.

## 2026-10-04 lifetime fix (reviewer-gpt-agent req_3b37d17e61374a6a)

The fast path now copies each host input into scheduler-owned pageable staging (per input copy) before ggml_backend_tensor_set_async: the source may be a pinned host buffer (truly async DMA) that the caller rewrites on the next set_inputs (e.g. the next prefill ubatch). The results above were measured before this fix; re-measure (queue-v5b-abba.sh) before final adoption.

## 2026-10-04 v5c re-measure (staged) and size cap

v5c ABBA x2 with pageable staging: decode ~8K 79.1 -> 86.5 t/s (+9.4%), ~64K 57.1 -> 61.5 (+7.8%), greedy identical 8/8 per depth, but prefill -2.3% / -2% (the earlier +3.7/+7.2% prefill came from the unsafe zero-copy path). The fast path is now limited to inputs <= 4 MiB (decode KQ mask at 240K for a 4-token verify is ~2 MB; prefill masks ~10 MB take the upstream path), so prefill returns to v4 behaviour. A pinned staging ring with copy-slot events could recover the prefill gain safely (future work).


## 2026-10-06 async-semantics clarification

ROCm HIP documents that hipMemcpyAsync with non-pinned host memory is performed synchronously. The current scheduler-owned std::vector staging is pageable, so its production decode gain must not be described as true H2D overlap. Its value is that the source lifetime is made safe while the scheduler avoids the old per-input destination-backend synchronization; the HIP pageable transfer itself may block the host.

This also explains why replacing the unsafe direct pinned source with pageable staging retained the decode win but lost the prefill gain. QFP16 now owns one bounded residual gate: measure whether large prefill input handling remains >=1 ms or >=3% of prefill wall time. Only then consider extending this same 1326 owner with a bounded pinned staging ring whose slots are protected by completion events from every consuming destination device. Otherwise retain the current <=4 MiB path and close the residual.
