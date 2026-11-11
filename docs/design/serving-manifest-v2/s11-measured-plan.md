# RSC11 S11: measured v2 planning and admission refit (inactive)

S11 provides `tools/bigcherry/serving/measured_plan.py:MeasuredPlanValidator`, the concrete injected `Supervisor.plan(layout, run_id)` implementation. It does **not** activate `bigcherry serve start`, change the v1 manifest, start GPU processes, or import any engine C++ patch. The real S5 engine and S10 v2 manifest writer are integrated once at S13. Card-free tests exercise the full algorithm through fakes, including the S12 supervisor/power reader.

## S11 responsibilities

1. Ask the injected `MeasurementSource.measure(layout, PlanBounds)` for an authoritative `PlanMeasurement`: the bc-serve-selected BDFs; real owner-map digest; planning cache key; engine-measured host embedding/unowned expert/staging/ngram/checkpoint/other bytes and KV bytes per token; intended context/concurrency/retention; current MemAvailable, cgroup headroom, pin availability; per-helper non-expert `worker_footprint.h` bytes; measured per-engine-GPU required and immutable VRAM bytes. These must be attached to provenance. No guessed defaults.
2. Derive host pool, host KV, helper budgets, pin and RAM admission through `layout.derive_resources()` (RSC03). The output and exact effective targets go to `ManifestWriter.write()`, which S10 must back with **one** native v2 writer and strict validator. Planning identity and owner map are constant through a calibration-only refit. Each attempt has its own immutable manifest.
3. Obtain a `AdmissionReport(available_bytes, required_bytes)` from `EngineValidator.validate()` using the S5 engine's pre-weight check. Persist an atomic `<run_id>.plan.json` receipt containing the inputs' provenance, derived budgets and measured room (including spare) or structured refusal.
4. On `EngineRoomRefused(RoomRefusal)`, retry **once**, not on stderr, timeouts, generic exceptions or post-weight failure. For host room, increase the host-ledger reserve by the measured shortfall and release enough *whole allocated MiB* of retained KV to make actual space; for pin room, release real KV without increasing the RAM margin. Both preserve the measured static host pool. If that much KV is unavailable, refuse with exact bytes to free. For GPU room, feed the exact observed claimable byte ceiling back to the engine measurement source and require an actual replanned engine GPU footprint below that ceiling. If the fixed GPU footprint exceeds claimable, refuse immediately. A repeated refusal, no-progress measurement, changed owner map/cache key or local RAM/pin budget failure is terminal, with the observed figures.
5. Return the validated `ServingPlan(manifest, ordered ProcessSpec tuple)` to the S6/S12 supervisor. Its existing checks govern lease, immutable manifest, power, processes, receipts and stop ordering. The planner never applies power.

This addresses the 2026-11-11 R9700 near-miss specifically: the engine reported **33,242,744,312 required** against **33,241,708,024 claimable**, a **1,036,288-byte deficit** (slightly under 1 MiB). A GPU retry must replan its *elastic* resident VRAM to at most the observed claimable bytes. Silently reusing the old GPU plan, or decreasing host KV alone without changing GPU usage, is not a retry.

## S5 wire contract (engine session; separate engine slice)

The S5 process adapter must recognise a dedicated nonzero admission exit status and read a durable, versioned JSON result, not parse human-readable stderr. The refusal record is exactly:

```json
{
  "schema_version": 1,
  "kind": "serving_admission_refusal",
  "phase": "before_weights",
  "reason_code": "gpu_room",
  "device_bdf": "0000:03:00.0",
  "available_bytes": 33241708024,
  "required_bytes": 33242744312,
  "allocations_still_to_come_bytes": 33242744312,
  "headroom_bytes": 0
}
```

`reason_code` is one of `host_room`, `gpu_room`, `pin_room`. `device_bdf` must be the selected engine GPU BDF for GPU room and JSON null otherwise. All byte fields are nonnegative integers; `required_bytes = allocations_still_to_come_bytes + headroom_bytes`, and a refusal requires `required_bytes > available_bytes`. The reported `available_bytes` is the **effective admission limit actually used** after live MemAvailable/cgroup/pin/GPU constraints, not a cached estimate. No weight payload is loaded on this path. S5 may emit an admitted result with the actual available, required and spare bytes for successful starts; the S13 adapter maps it to `AdmissionReport`.

The S5 transport/exit-code value is reserved for the engine S5 slice to define; S11 deliberately does not hardcode an exit number or use text fallbacks. A malformed or missing typed record is fatal, never retryable. S5 also needs a measured elastic/fixed VRAM split and must honour an exact per-BDF GPU ceiling in the re-measurement path; this is essential to solve the near-miss and must not be implemented as a tolerance.

## S10/S13 responsibilities and activation barrier

- S10: implement `MeasurementSource` against native bc-serve/engine `--plan-memory` with one owner map, strict provenance and a stable plan-cache key; implement `ManifestWriter` through v2 only. Include the refit margin in `spec.calibration.host_ledger_margin` (source measured/cache, evidence, added bytes), derive plan resources from the exact effective `Targets` supplied by S11, and authenticate the produced manifest/digest. Publish the engine's rederived GPU elastic placement and any host-KV migration under `PlanBounds.gpu_ceiling`; no static fallback.
- S13: implement a real `EngineValidator` that runs S5's pre-weight admission while under the serving lease and before the S12 power transaction/session start, with matching helper owner sets; return a structured report or `EngineRoomRefused`. Any resource-plan change requires a fresh manifest and S12 power/digest preflight. **Also handle a new S5 structured refusal during the final actual engine start** (between admission probe and live start) by stopping/verifying all children exited, restoring/reapplying power safely where applicable, and invoking the same bounded refit path. The S11 preflight alone does not eliminate time-of-check/time-of-use changes.
- RSC10 plan cache: persist measured host-ledger margin and start-room samples with provenance under the stable planning key, using the largest shortfall in a calibration window to allow later downward refits; do not let a single successful start drop a cached margin. The current S11 run receipt is durable evidence for that cache but does not implement windowed cache policy.
- Until S13, production v1 behaviour remains byte-identical; there is no v1 adapter, no legacy compatibility shim and no GPU use by S11 tests.

## Card-free gate

Run `pytest -q -o pythonpath=tools tools/tests/serving/test_measured_plan.py tools/tests/serving/test_supervisor.py tools/tests/serving/test_power_plan.py tools/tests/cli/test_serve_cli.py tools/tests/campaign/test_rsc03_resources.py`, plus repository CI. The tests include the exact R9700 near miss, fit-first, host and pin refit, terminal second refusal, a no-progress engine, immutable fixed-GPU terminal, insufficient KV, changed planning identity, hostile malformed records, non-S5 failure and integration with the existing S12 power reader.
