# RSC11 S13 P1 — inactive native publication slice

This slice provides an engine-side, GPU-free publication boundary
(`publish_p1_measurement(P1Native)`) and strict S11 Python consumer
(`P1MeasurementSource`). It is **not** a production S13 cutover. The shipped
`bc-serve`, v1 manifest, active launch and GPU paths are unchanged.

The C++ publisher takes the engine's actual `radiance --plan-memory`
`EnginePlan`, actual `OwnerMap`, measured per-rank GPU demand, per-helper
native lane/shape/stride inputs, and measured live host room. Helper footprint
components are computed through the worker's **shared** `worker_footprint.h`
formulas. Engine host consumer names and checkpoint geometry are read from
the plan. KV bytes per token are the exact
`engine_plan.kv_one_session.bytes / context_target_tokens`, with positive
integral division required: calibration target MiB never becomes a KV density.
The record carries a planning key, provenance, owner-map digest, scope,
all v2 `MeasuredResourceInputs`, per-rank demand and S11 targets. Missing,
duplicate, non-integer or out-of-scope values are refused. P2 supplies the
qualified host-margin record; this code never seeds one.

The Python parser strictly rejects malformed JSON, duplicate members,
fractions, missing fields, owner/scope disagreements, invalid worker
component totals, nonselected helper/engine BDFs and incomplete live room.
It constructs S11 `PlanMeasurement`, retaining exact resource inputs and
scope for the S10 v2 writer. A `P1MeasurementSource` accepts an injected
native command builder and makes a fresh invocation on every refit.

## Remaining P1 integration, deliberately not claimed by this slice

1. Connect the native P1 inputs to bc-serve's **actual** model-directory,
   owner-map, engine-plan, selected helper and live host measurement paths.
   Expose one explicit inactive native measurement command; do not derive
   any value from a v1 manifest or `host_kv_tier.target_mib`. The current
   engine plan does not itself publish unowned expert bytes, measured
   resident-other host bytes, live effective cgroup/pin headroom, or
   per-rank GPU fixed/required bytes as a single S8 record.
2. Wire that command to S11's `P1MeasurementSource` in the S13 adapter.
   Re-key/re-measure on model/build/owner/layout/inventory/targets changes,
   not just GPU shortage. S10's v2 writer must consume `resource_inputs`
   as published, compare with engine `--plan-memory`, and fail closed.
3. Use only a scoped, time-valid P2 calibration record (P2 #855); no
   synthetic test measurement is hardware admission evidence. Under a GPU
   lease the lead must confirm real plan KV divisibility, every worker
   footprint component and live RAM/cgroup/pin readings on supported
   engine/helper and TP layouts before S13 activation.

The synthetic C++ and Python tests test validation and arithmetic only.
