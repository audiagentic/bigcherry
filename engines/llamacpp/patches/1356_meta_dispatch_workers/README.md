# 1356_meta_dispatch_workers

QFP41 persistent per-device graph-submission workers for the Meta tensor-split backend.

Default: **off**. Enable with `BIGCHERRY_META_DISPATCH_THREADS=1`.

Build experiment `meta-dispatch-workers` and run:

```bash
BIGCHERRY_PATCH_TRACE=1 FIDELITY=1 \
AB_ENV="BIGCHERRY_META_DISPATCH_THREADS=1" \
tools/lab/flash-next/queue-env-ab.sh qfp41-workers <build-run-id> 8192 24576 98304
```

A = production/off; B = workers/on. Require the patch-hit marker in B, identical greedy md5 at every depth,
neutral acceptance/decode control, and complete separation before claiming a prefill win.

Repeatability/stress:

```bash
BIGCHERRY_PATCH_TRACE=1 AB_ENV="BIGCHERRY_META_DISPATCH_THREADS=1" \
tools/lab/flash-next/queue-env-ab.sh qfp41-repeat1 <build-run-id> 8192 24576 98304
BIGCHERRY_PATCH_TRACE=1 AB_ENV="BIGCHERRY_META_DISPATCH_THREADS=1" \
tools/lab/flash-next/queue-env-ab.sh qfp41-repeat2 <build-run-id> 8192 24576 98304
DECODE_N=2048 BIGCHERRY_PATCH_TRACE=1 AB_ENV="BIGCHERRY_META_DISPATCH_THREADS=1" \
tools/lab/flash-next/queue-env-ab.sh qfp41-stress <build-run-id> 98304
```

The two qualification runs must reproduce greedy md5s and performance ordering; the deep stress run must complete
without hang/crash. Run a host-side ThreadSanitizer build where the HIP toolchain permits it; sanitizer inability is
recorded, not silently waived.
