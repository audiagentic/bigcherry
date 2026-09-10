# Testing — NRO01 Q8 AllReduce wire

## Static gates

1. `patch-lint --json` and package discovery.
2. Dependency closure must include `1001_hip_internal_allreduce`.
3. Apply on pin `b10705`; second apply must be `already-applied`.
4. HIP build with threshold unset/0 must behave exactly as pre-NRO01.
5. Non-HIP compile must remain valid because the internal code remains behind existing backend guards.

## Required correctness fixture before live dispatch

Generate two independent FP32 rank arrays and compare Q8 reduction against CPU FP32 reference for: all-zero, nonzero asymmetric ramps, alternating signs, random normal, high dynamic range, repeated extrema, and element counts `{1,31,32,33,255,256,257}` plus real collective sizes. Validate both ranks and verify padded Q8 tail bytes cannot affect valid elements.

Record max absolute error, zero-safe relative error, MSE/NMSE and any model-level quality metric. Tolerance must be registered before throughput measurements.

## Hardware campaign prerequisites

Only after the fixture passes: wire Q8 into the copy path; add activation marker/counters for logical bytes, Q8 blocks and wire bytes; compare exact/BF16/Q8 on dual gfx1100 across decode and prefill. Prefill is mandatory because exact internal already has a known large-message regression.

`PASS` requires correctness + activation + paired performance. An environment variable or faster wall clock alone is not activation evidence.
