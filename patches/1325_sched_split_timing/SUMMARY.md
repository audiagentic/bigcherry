# 1325_sched_split_timing

**Status:** untested
**Plan item:** QFP16

## What it does

Diagnostic only. With `BIGCHERRY_SUBMIT_TIMING=1` every scheduler split logs `BIGCHERRY_SCHED_SPLIT` with its backend,
input count, input-handling host time (synchronize/copies between backends) and async-compute submit host time
(synchronous for CPU splits). Attributes the part of the target's ~5 ms graph_compute that is outside the meta backend
(1320 measured ~2.2 ms inside it).
