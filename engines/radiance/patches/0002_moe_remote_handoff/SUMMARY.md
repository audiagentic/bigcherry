# 0002_moe_remote_handoff

**Status:** untested
**Plan item:** RAD01

## What it does

Adds an optional whole-expert MoE hand-off ABI. When the mailbox hook is registered and the row policy admits a pass, helper-owned routes are masked from the local grouped GEMMs and published to the GPU helper. The normal MoE gather materializes the local token output, `moe_mailbox_add` then adds the helper rows with their published BF16 route weights, and the existing standalone tensor-parallel all-reduce runs afterward. HyperConn fused gather/reduce forms are disabled only for connections with a declared MoE gather source on those remote passes. With no mailbox or an excluded row count, the existing fused path is unchanged.

## Why

The seam allows a helper GPU to compute cold routed experts rather than making the serving ranks repeatedly consume host-resident expert weights across the link.

## Evidence

The motivation is documented on `feat/rad11-gfx1100-bootstrap` in `engines/radiance/kernels/README.md`, section “Flash-Next: radiance's own prefill stager, switched on”, which records cold-expert transfer pressure and the R9700 helper context. That branch does not contain mechanics/correctness/KL/A-B qualification for this remote-expert hand-off, so the package remains `untested`.

## Upstream

Local BigCherry change against Radiance 1.3.0, commit `89cee7ce414486c717aa2085c99e1f522e4c0ea6`; source diff: `engines/radiance/engine-patches/0002-moe-remote-handoff.patch` on `feat/rad11-gfx1100-bootstrap`.
