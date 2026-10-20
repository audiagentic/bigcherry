# moe-remote: remote-MoE hand-off helper and mailbox plugin

## Summary

The remote-MoE hand-off (RAD02, RCS04): a separate `r3_moe_worker` process per
card owns whole routed experts and computes them for the engine over a pinned-
host mailbox with GPU-side flags; `libr3moe` is the engine-side mailbox plugin
that fills the ops engine patch `rad0002_moe_remote_handoff` declares. Split out
of `libr3`, which keeps the engine kernel overrides.

## Outcome

`experimental`: `R3_MOE_WORKER` and `R3_MOE_WORKER_DEVICES` are measured in the
helper-ab A/B harness (`tools/lab/radiance/helper-ab.json`) but not yet the
production default; see the Flash-Next model page
(`docs/reference/engines/radiance/models/qwen3.8-flash-next.md`) for the
current best numbers using this helper layout.

## Detail

Settings and status vocabulary: see `component.toml` (`R3_MOE_WORKER`,
`R3_MOE_WORKER_DEVICES` harness-only, `R3_MOE_WORKER_SHM` diagnostic shm
segment name, `R3_MOE_WORKER_CPU` forces the slice-2a fp32 host reference path
instead of gfx1201 compute). Offline tests: `r3_moe_shm_test`,
`r3_moe_backend_test`, `r3_moe_fp8_ref_test` (ctest in the component build
tree). Owner rule (2026-10-13): the R9700 is compute in this layout, not
storage — lookup-type side work (see `ngram-helper`) goes to host RAM/CPU
instead.
