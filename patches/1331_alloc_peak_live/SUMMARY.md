# 1331_alloc_peak_live

**Status:** untested
**Plan item:** QFP17

## What it does

With `BIGCHERRY_ALLOC_PEAK=N`, the graph allocator tracks live bytes per compute buffer and, at each buffer's peak,
logs the node being allocated, the N largest tensors live at that moment and live bytes per op
(`BIGCHERRY_ALLOC_PEAK ...` lines at sched_reserve). This is what sets the compute buffer size; 1329 only ranks
the largest tensors overall, which misled 1330 (removing a large short-lived mask left the ub1024 buffer unchanged).
Fragmentation is not modelled; compare with the `compute buffer size` line. No behaviour change when unset.
