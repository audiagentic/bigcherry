# 0830_split_reduce_telemetry framework validation

Instruments the SPLIT_REDUCE allreduce boundary (`ggml-cuda.cu`) to record
which reduction provider (RCCL/meta) actually handled a given reduction and
the handoff between providers, at the existing allreduce context rather than
a separately reimplemented path. Needed to verify, from real telemetry
rather than assumption, which reduction path a given multi-GPU run actually
used (HI58).

This local framework adapter has no Experiment Contract: it checks build
plumbing, not a claimed kernel speedup. Historical `validated` state is not
current qualification.

Validation is the universal `apply`/`build` checks only -- no patch-specific
custom check exists for this package. Passing proves the patch applies
cleanly to the pinned upstream source and the resulting tree compiles; it
does NOT prove the recorded provider/handoff telemetry is accurate for any
given multi-GPU run.

## Upstream / provenance

Local design, part of this project's own telemetry work (HI58).
