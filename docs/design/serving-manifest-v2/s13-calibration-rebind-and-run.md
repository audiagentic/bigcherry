# RSC11 S13: measured S8 runs and explicit build re-binds

This is a **card-free preparatory integration**. It does not activate v2.
Production has only the existing v1 manifest until the single atomic S13
reader/writer/launcher cutover. No synthetic fixture is hardware qualification.

## Hardware measurement run files (new producer)

`tools/bigcherry/serving/s8_lab.py` is a lab orchestrator. It invokes an
**actual hardware runner**, checks the GPU lease on every arm, records the
runner's applied-value/readback claim, verifies closed observations and explicit
numerical gates, archives per-arm stdout, stderr, raw evidence and full results,
and emits `s8_calibration_measurement.json` accepted by the P2
`emit_from_run` record store. No S8 tunable default is selected. An operator
must supply a native P1 `bc_serve_v2_measured_inputs` publication, actual
candidate values, a qualified performance/latency gate file, the hardware
runner, and an explicit evidence validity end time.

Example (the lead substitutes **real** candidate values, runner, scope,
end time, and gate thresholds based on the objective and model; none below are
calibration settings):

```sh
agent-lease acquire gpu -- \
  python3 tools/lab/radiance/s8-calibrate.py \
    --key handoff.small_rows \
    --native-measurement ${BC_WORK_ROOT}/runs/real-p1/native.json \
    --runner ${BC_WORK_ROOT}/runs/real-p1/s8-hardware-runner \
    --candidate "${CANDIDATE_A}" --candidate "${CANDIDATE_B}" \
    --gates ${BC_WORK_ROOT}/runs/real-p1/gates.json \
    --objective throughput --valid-until-unix-s "${VALID_UNTIL}" \
    --reason "ABBA same-build worker row-shape and end-to-end hardware gate" \
    --out ${BC_WORK_ROOT}/runs/s8-small-rows-verified \
    --store ${BC_WORK_ROOT}/s8-calibration-records
```

The runner is an executable the lead supplies. For each arm the producer
calls it as `RUNNER [--runner-arg ...] --s8-key KEY --s8-scope PATH
--s8-output PATH --s8-value VALUE` (for the margin: the last argument is
`--s8-measure-host-room`). It must **actually apply and read back** the named
candidate on selected BDFs, run the specified workload under the lease,
and write a JSON observation plus a nonempty raw evidence file in the same
output directory. The current v1 launcher cannot itself apply v2 S8 values:
it must not be used as a pretend v2 runner. Hardware driver integration
and GPU qualification remain the lead's work; this orchestrator never claims
to supply a tested hardware runner or qualification on its own.

For a handoff key the exact observation members are:

```json
{
  "schema_version": 1, "kind": "s8_hardware_observation",
  "key": "handoff.small_rows", "candidate": 2, "applied_value": 2,
  "scope": {"...": "exact native P1 scope"},
  "workload_digest": "<64 lowercase hex chars>",
  "output_correct": true, "health_ok": true, "thermal_ok": true,
  "effect_window_ok": true, "migration_stable": true,
  "metrics": {
    "throughput_milli_tok_s": 1000, "tail_latency_us": 1000,
    "overflow_count": 0, "histogram_write_us": 0, "max_heat_age_ns": 0
  },
  "raw_evidence": "arm-000.hardware.json"
}
```

Numbers here illustrate the **wire shape only**: they are neither proposed
settings nor admission gates. The actual runner must use the native scope,
real candidate and measured values, with raw evidence. The gate file contains
`min_throughput_milli_tok_s` and `max_tail_latency_us` (positive, explicit);
the histogram keys also require `max_histogram_write_us`, and live cadence
also requires `max_heat_age_ns`. Required output correctness, health, thermal,
effect-window, migration stability, no overflow, and workload digest equality
are unconditional gates. The orchestrator uses matched forward/reverse ABBA
sweeps (at least four clean pairs) and selects from only the tested candidates
by the operator's throughput or latency objective. For kernel mode, only
explicit 0/1/2 candidates are admitted. The independent dispatch-capacity
ceiling is still an S8 measured input, never guessed by this command.

For `host_ledger_margin.mib` there are no handpicked candidates or
throughput gates. The hardware runner must perform repeated pre-weight
admission observations and write `s8_host_admission_observation` rows with
the exact P1 scope, same `planning_key`, `required_bytes`,
`available_bytes`, `planned_reserve_bytes`, `shortfall_bytes` (the latter
equal to max(0, required - available)) and a raw file. The new margin is
`prior qualified margin + ceil(max observed shortfall/MiB)`, or just the
measured shortfall when no prior margin exists. If refitting, give
`--prior-margin-record PATH` and the producer validates its exact scope and
freshness. A GPU VRAM shortage is not a host margin sample. Failure or missing
raw evidence leaves no qualifying run file.

These are **method constraints**, not guaranteed hardware evidence: the
runner must itself capture correct device selections, sample shape/ABBA
health and raw performance data. P2 has no hardware verification hook and
cannot authenticate a fabricated runner, so review the raw run before
using `--store`. No test in this slice accesses the GPU.

## Re-bind a qualified record after an unaffected build bump

The record store remains strictly scoped to the model, engine/helper build
digests, layout, inventory, owner map, context, retention and KV geometry.
A source record is **not** a wildcard across new builds. The re-bind command
uses an original P2 record, a new native P1 measurement, a code revision diff,
and an explicit reason. It writes an immutable
`STORE/rebinds/<content-sha256>.json` with original-record SHA-256,
old/new exact scopes and build digests, Git commit SHAs, all changed paths,
the reason and UTC timestamp. It neither rewrites the original hardware
run nor extends its validity.

```sh
python3 tools/lab/radiance/s8-rebind.py \
  --source-record ${BC_WORK_ROOT}/s8-calibration-records/<original>.json \
  --target-measurement ${BC_WORK_ROOT}/runs/new-build/native.json \
  --store ${BC_WORK_ROOT}/s8-calibration-records \
  --repo . \
  --old-rev "${OLD_SOURCE_COMMIT}" --new-rev "${NEW_SOURCE_COMMIT}" \
  --reason "Only engine/helper build digests changed; new P1 and S5/S11 admission must still pass"
```

The Git diff is an **audit record, not a source-path allowlist**. A
build-only P1 scope change is explicitly re-bindable even when Radiance,
helper, kernel, build tooling or other runtime sources changed. An empty
changed-path set, including the **same source commit** with different engine
or helper build digests, is also permitted (for example, a toolchain rebuild). The tool still records exact old/new
build digests, source commits, **all** changed paths and a nonempty rationale,
hashes the immutable proof and binds the exact source hardware record. The
original evidence validity window is never extended.

The safety authority is the mandatory fresh native P1 measurement and S8
measured geometry check at **each** start, followed by S5 pre-weight
admission and S11's bounded two-attempt refusal/refit. In particular, a
calibrated handoff maximum cannot exceed the new engine's measured dispatch
capacity; host/pin/GPU plans must fit new live measured room. If the fresh
P1 owner/context/retention/KV measurements fall outside the record's
original exact measured scope, or the S8 capacity or terminal S5/S11
admission gate fails, startup **refuses**, naming the re-bound source record
and asking for hardware re-measurement. A successful bounded S11 refit may
proceed; a rejected attempt never permits a power or launch bypass. No new
quantitative tolerance or previously unmeasured operating envelope is
inferred from a filename or from the build bump.

A change of model/layout/inventory/owner-map/context/retention/KV scope
**always** requires new evidence, regardless of changed paths or rationale.

`QualifiedP1MeasurementSource` reads the record store on *every* attempt.
If a current-scope P2 record exists, it is selected directly; otherwise one
exact, hash-verified rebind receipt can qualify a still-fresh original record.
The emitted C++ S8 evidence ID retains the original raw run plus
`rebinds/<receipt>` provenance. No receipt, multiple matches, expired
original record, malformed proof, P1 margin mismatch or changed non-build
scope => explicit refusal. Hardware start failures request re-measurement
with the exact record identity; source path changes do not automatically do so. The source has to provide this rebound evidence ID to
the native P1 margin publisher; no historical literal is inserted.

**Cutover still pending:** native P1 publication from live bc-serve,
S10/strict v2 production writer/readers, engine/helper and Python launch
adapter integration, and runtime live-policy decision/observation receipts.
The lab runner cannot qualify values until it can apply them on the actual
runtime and measure them under a GPU lease.
