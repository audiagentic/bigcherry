# rad0002_moe_remote_handoff

**Status:** untested
**Plan item:** RAD01

## What it does

Adds an optional whole-expert MoE hand-off ABI. When a hook is registered and `R3_MOE_REMOTE_MIN_ROWS` admits the step, host-pool expert routes are removed from local grouped GEMMs, computed by the hook, copied into an engine-owned BF16 partial, and merged before the existing tensor-parallel all-reduce. With no hook or the default threshold, the existing path remains selected.

## Why

The seam allows a helper GPU to compute cold routed experts rather than making the serving ranks repeatedly consume host-resident expert weights across the link.

## Evidence

The motivation is documented on `feat/rad11-gfx1100-bootstrap` in `engines/radiance/kernels/README.md`, section “Flash-Next: radiance's own prefill stager, switched on”, which records cold-expert transfer pressure and the R9700 helper context. That branch does not contain mechanics/correctness/KL/A-B qualification for this remote-expert hand-off, so the package remains `untested`.

## Upstream

Local BigCherry change against Radiance 1.3.0, commit `89cee7ce414486c717aa2085c99e1f522e4c0ea6`; source diff: `engines/radiance/engine-patches/0002-moe-remote-handoff.patch` on `feat/rad11-gfx1100-bootstrap`.
