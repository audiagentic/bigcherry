# S8 calibration evidence (inactive P2 contract)

`tools/bigcherry/serving/calibration_store.py` is the S13 preparation
boundary for S8's `S8Calibration` records. It persists exactly the members
in `engines/radiance/components/serving/v2/s8_rules.h`, not a parallel set
of memory defaults. No startup path reads this directory until S13.

A measurement producer writes one JSON run output with:

- `schema_version=1` and `kind=s8_calibration_measurement`.
- `key`, `scope` (all six SHA-256 model/build/layout/inventory/owner
  digests and positive context, retention, engine-measured KV bytes/token),
  `raw_run`, `evidence_id`, `method`, `source`.
- `measured_at_unix_s`, `valid_until_unix_s`, `selected_value`.
- Nonempty `samples`, each `{ "candidate": <measured integer>,
  "evidence_ref": "<raw sample reference>" }`. The selected value must
  appear among the independently recorded candidates.

There is **no command-line selected-value override**. From the repository
checkout (with the package on PYTHONPATH), emit with:

```bash
PYTHONPATH=tools python -m bigcherry.serving.calibration_store emit \
  --run path/to/measured-run.json --store work/serving/calibrations
```

The generated record is an immutable JSON file addressed by the canonical
SHA-256 of the supplied run. `select_records(store, scope=...,
required_keys=..., now_unix_s=...)` requires one fresh, exactly matching
scope for every requested S8 key; stale, future-dated, ambiguous, malformed
and unmeasured records are rejected, with no fallback. Unknown scoped files
are not used. The S13 bridge will map selected records to C++ `S8Calibration`
and pass them into `resolve_s10`/`resolve_s8`. The native resolver remains
responsible for applying the handoff/host admission rules.

Every time the model, engine/helper builds, layout, inventory, owner map,
context/retention target or measured KV geometry changes, collect new
evidence and emit new records. Hardware owners must supply the run,
raw evidence and expiry; **this PR seeds no calibration values**. These
synthetic card-free tests are not qualification or evidence for the supported
production layouts.
