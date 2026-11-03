# moe-remote: remote-MoE hand-off helper and mailbox plugin

## Summary

A separate `r3_moe_worker` process per helper card owns routed experts and computes their rows over a pinned-host mailbox with GPU-side signals. `libr3moe` is the engine-side plugin for the production `rad0002_moe_remote_handoff` hooks; `libr3` still owns the engine's kernel overrides.

## Outcome

The **engine hand-off patch** is validated in the Radiance production `serving-core` set; the component's individual settings retain their own states in `component.toml` (some are experimental/diagnostic rather than production defaults). See the [Flash-Next results](../../../../docs/reference/engines/radiance/models/qwen3.8-flash-next.md) for measured helper deployments.

## Detail

For live placement control, read [ADMISSION.md](ADMISSION.md): the per-step **admission ledger**, GPU consumption proof, safe lane/helper **drain**, placement quiescence and reopen, and same-placement **epoch cycles**. Draining gates admission and requires proof that the old placement has finished before a new epoch can begin; epoch cycles alone do **not** move expert ownership or migrate weights.

`R3_MOE_WORKER_EPOCH_CYCLE_S` is a **diagnostic-only** environment variable: zero (default) disables periodic same-placement epoch cycles. A positive representable integer period in seconds exercises the drain/reopen protocol; it is not a normal serving policy or a live expert migration setting. See ADMISSION.md for semantics, limits and validation evidence.

Worker settings are defined in `component.toml`. `R3_MOE_WORKER` enables the helper; `R3_MOE_WORKER_DEVICES` selects devices in the benchmark harness; `R3_MOE_WORKER_SHM` names the diagnostic shared-memory segment; `R3_MOE_WORKER_CPU` uses the CPU reference computation. Card-free component tests include `r3_moe_shm_test`, `r3_moe_backend_test`, `r3_moe_fp8_ref_test`, and the admission/epoch tests.
