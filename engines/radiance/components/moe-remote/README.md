# moe-remote: remote-MoE hand-off helper and mailbox plugin

## Summary

A separate `r3_moe_worker` process per helper card owns routed experts and computes their rows over a pinned-host mailbox with GPU-side signals. `libr3moe` is the engine-side plugin for the production `rad0002_moe_remote_handoff` hooks; `libr3` still owns the engine's kernel overrides.

## Outcome

The **engine hand-off patch** is validated in the Radiance production `serving-core` set; the component's individual settings retain their own states in `component.toml` (some are experimental/diagnostic rather than production defaults). See the [Flash-Next results](../../../../docs/reference/engines/radiance/models/qwen3.8-flash-next.md) for measured helper deployments.

## Detail

For live placement control, read [ADMISSION.md](ADMISSION.md): the per-step **admission ledger**, GPU consumption proof, safe lane/helper **drain**, placement quiescence and reopen, and same-placement **epoch cycles**. Draining gates admission and requires proof that the old placement has finished before a new epoch can begin; epoch cycles alone do **not** move expert ownership or migrate weights.

`R3_MOE_WORKER_EPOCH_CYCLE_S` is a **diagnostic-only** environment variable: zero (default) disables periodic same-placement epoch cycles. A positive representable integer period in seconds exercises the drain/reopen protocol; it is not a normal serving policy or a live expert migration setting. See ADMISSION.md for semantics, limits and validation evidence.

Worker settings are defined in `component.toml`. `R3_MOE_WORKER` enables the helper; `R3_MOE_WORKER_DEVICES` selects devices in the benchmark harness; `R3_MOE_WORKER_SHM` names the diagnostic shared-memory segment; `R3_MOE_WORKER_CPU` uses the CPU reference computation. Card-free component tests include `r3_moe_shm_test`, `r3_moe_backend_test`, `r3_moe_fp8_ref_test`, and the admission/epoch tests.

## RAD46 gfx1030 rows replay (experimental)

The RDNA2 W4 rows fast path narrows the scale-multiplied operands to fp16,
which can change small products before the fp32 reduction. Two opt-in
large-lane, gfx1030-only alternatives preserve the validated output-lane
arithmetic: `R3_MOE_WORKER_OUTPUT_LANE=1` (reference) and
`R3_MOE_WORKER_QUAD_DPP=1` (four-lane ordered fold; takes priority if both
are set). The kernel uses unscaled fp16 inputs, reset dot8 pairs, an ordered
K64 sum, then f32 weight and activation scales over K128. Other formats,
devices, lanes and incomplete K128/N128 shapes retain the original dispatch.
Switches are read at worker startup and refreshed at replay-sweep variant
boundaries, never from the request path.

These are diagnostic alternatives, not a default: the accurate quad fold
matched reference hashes in the existing captures, but prior large-request
profiles still showed gate/up slower than the original. Do not enable for
production until leased kernel profiles and the lead's quality/throughput
acceptance checks pass.

RAD46 quad-v2 experimental variant: set `R3_MOE_WORKER_QUAD_V2=1` to
select an independent gate/up instantiation with `kQuadDppRB / 2` (8)
rows per batch rather than 16, reducing live f32 accumulator/first-group
values and LDS staging; down still uses the established 16-row quad kernel.
Each row's dot8 reset, ordered four-dot16 fold, two-scale K128 fold, and
single BF16 narrow remain unchanged, so accurate hashes should match
`R3_MOE_WORKER_OUTPUT_LANE=1` exactly. v2 overrides the other two
experimental flags; it is not enabled by default. This is an unmeasured
performance candidate until lead GPU replays validate hashes and timings.
