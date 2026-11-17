# RCS13 / PR #946 — Run 8 admission after #1058

2026-11-17 UTC. **NOT CLEARED — zero end-to-end serving samples.**

The two official full `radiance:stock:linux-radiance` campaigns from main
`e28734f097fe7c0ff59415ecfe0494271a1e22a0` and PR #946 `778c7d701b10ec18bccfda3d2423e90272558901` succeeded with exactly
the seven RCS13 files different. Same upstream source slice, patch tree,
pin v1.17.0, 30 validated / 0 pending patches, effective configure except
equivalent ccache paths, and engine + libr4d SHA-256. Both installed header
and libr4d ABI are 35. See `build-attestations.json`.

The normal Python `RadianceAdapter.write_serving_manifest` produced the
baseline's strict-v2 measured manifest. Host-pool preflight 32005 MiB was
reconciled to the engine-measured 32301 MiB, with consistent measured owner
map (12641 experts). **This baseline was a measured-plan cache HIT**: the
invocation did not start its own measuring helper/engine.

The candidate performed **two real measuring starts** (a cache miss followed
by host-pool remeasurement, 32005 to 32301 MiB), with its source-matched
helper starting and reporting W4H2 *enabled*. The final native
`bc-serve --write-manifest --measured-plan` failed with exit 2:
`BC_CFG_ADMISSION_REFUSED (RAD_E_NOMEM, -6)`. The measured owner map
required **30555 MiB** helper expert VRAM; the sampled live GPU capacity
after the helper footprint allowed only **1826 MiB**, a **28729 MiB**
shortfall. Native `serving/v2/s10_resolver.cpp:355-366` correctly refuses
to silently shrink ownership after measurement. Inputs and S8 ceiling
calculation: `serving/v2/s10_inputs.cpp:110-127` and
`serving/v2/s8_rules.cpp:244-275`. Python adapter propagates error at
`tools/bigcherry/engine_adapters/radiance_plan.py:186-264`.

The CPU-only `cpu_admission_repro.py` replays the fail-closed inequality
using the original immutable measuring draft and captured validator stderr.
It was run successfully on the test host. **It does not independently
reconstruct historical sysfs VRAM usage** or establish why live capacity
was so low; this needs investigation under fresh GPU telemetry. The helper
log showed zero served requests, so an enabled W4H2 flag is **not** executed
dynamic-LDS kernel evidence.

No production A/B/B/A arm was launched. Decode engine ms/step,
prefill/TTFT, MTP drafted/accepted totals, exact output hashes, live
K=2560 W4H2 dispatches, and occupancy remain unmeasured. Do not claim
speedup, equality, or a kernel occupancy result. Lease was released and no
KFD processes remained. No commit was made to #946.

The raw detailed logs, measuring draft, source-specific binary receipts,
and engine plan are retained in the run-owner's scratch checkpoint and
not embedded here, to avoid machine-specific paths, secrets, or build files.
