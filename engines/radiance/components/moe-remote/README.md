# moe-remote: remote-MoE hand-off helper and mailbox plugin

## Summary

A separate `r3_moe_worker` process per helper card owns routed experts and computes their rows over a pinned-host mailbox with GPU-side signals. `libr3moe` is the engine-side plugin for the production `rad0002_moe_remote_handoff` hooks; `libr3` still owns the engine's kernel overrides.

## Outcome

The **engine hand-off patch** is validated in the Radiance production `serving-core` set; the component's individual settings retain their own states in `component.toml` (some are experimental/diagnostic rather than production defaults). See the [Flash-Next results](../../../../docs/reference/engines/radiance/models/qwen3.8-flash-next.md) for measured helper deployments.

## Detail

For live placement control, read [ADMISSION.md](ADMISSION.md): the per-step **admission ledger**, GPU consumption proof, safe lane/helper **drain**, placement quiescence and reopen, and same-placement **epoch cycles**. Draining gates admission and requires proof that the old placement has finished before a new epoch can begin; epoch cycles alone do **not** move expert ownership or migrate weights.

`spec.experiment.placement.epoch_cycle_s` of the serving manifest (lab profile only) is **diagnostic-only**: absent (the default) disables periodic same-placement epoch cycles. A positive representable integer period in seconds exercises the drain/reopen protocol; it is not a normal serving policy or a live expert migration setting. See ADMISSION.md for semantics, limits and validation evidence.

The helper reads its whole configuration from the serving manifest bc-serve writes (`r3_moe_worker --manifest FILE --role helperN`; the `--measuring` flag is the measuring start, which takes the unmeasured manifest): the shared strict v1 reader (`serving/v1`) checks it before any GPU work, and `serving/v1/helper_settings.h` is the flat projection the worker code uses. No helper `R3_*` environment variable is read; the one environment write is `GPU_MAX_HW_QUEUES`. The live placement policy is `spec.handoff.placement.live_policy`, the forced schedule `spec.experiment.placement.forced`. Setting descriptions and defaults are in `component.toml`. `R3_MOE_WORKER` enables the helper; `R3_MOE_WORKER_DEVICES` selects devices in the benchmark harness; `R3_MOE_WORKER_SHM` names the diagnostic shared-memory segment; `R3_MOE_WORKER_CPU` uses the CPU reference computation. Card-free component tests include `r3_moe_shm_test`, `r3_moe_backend_test`, `r3_moe_fp8_ref_test`, and the admission/epoch tests.

## RAD46 gfx1030 accurate W4 rows

W4 rows use **accurate arithmetic by default**. The old large-lane kernel
narrowed scale-multiplied operands to fp16 before accumulation, affecting
small products and failing the tail quality gate. The production W4 rows
path uses unscaled fp16 dot8 terms, an ordered four-dot16 K64 sum, f32
weight scales, and an f32 K128 activation scale before one BF16 narrow.

Dispatch uses the already-known request row count and selects **by stage**:
output-lane for at most 309 rows in gate/up and at most 922 in down;
quad-DPP (RB16) above each crossover. These boundaries are calculated
from the signed measured quad-minus-output-lane latency differences at
64 and 2048 rows in `rad46_rows_policy.h`, not arbitrary tuning switches.
The W4 rows path requires complete K128 tiles and fails closed otherwise;
it does not fall back to the old inaccurate kernel.

For kernel-only investigation, `--replay-sweep` alone accepts
`RAD46_SWEEP_VARIANT` values `stock`, `output_lane`, `quad`,
`quad_v2`, `production`, `rowlane16`, and `rowlane8`. A separate
`moe_lane_rows_stock_replay.h` isolates the historical incorrect kernel
for comparison only, never normal serving. `rowlane16` and `rowlane8`
are unmeasured gate/up candidates: quartet lanes own different routed
rows, share packed weight words via quad shuffles, and construct each
row's four K64 partials in K order without per-row DPP reductions.
Their down stages use production dispatch. New candidate hash identity,
per-stage speed, and end-to-end KLD/throughput remain measurement gates.
