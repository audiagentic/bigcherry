# rad0002_moe_remote_handoff

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

## Status 2026-10-14: evaluated, off by default (RAD03)

The R9700 expert helper (libr3 `R3_MOE_HELPER`, default off) and this hand-off are kept for later work, not in any
production set. Evidence on Flash-Next, two XTX + R9700, 4-document KL (64-token steps):

| Arm | KLD mean | top-1 |
|---|---|---|
| helper off | 0.000 | 100% |
| `R3_MOE_HELPER_SHADOW=1` (helper rows compared, engine rows used) | 0.000; every handed-off layer within rel_l2 2.4e-4 | 100% |
| `R3_MOE_HELPER_MASK_ONLY=1` (mask + rescatter + add, engine's own captured rows) | 0.000 | 100% |
| helper on (helper rows added) | 0.0616 | 88.8% |
| `R3_MOE_HELPER_POISON=zero` (remote routes dropped) | 0.745 | 65.3% |

Open question: the add reads helper rows that shadow finds equal to the engine's, yet only those rows (not the
engine's, through the identical add) lose quality, deterministically to the last digit across builds. Layers with
protected experts are kept local by the hook. Speed (decode 26.5 -> 47.9-50.5 tok/s on single loads) is not
attributable until quality passes. Lab scripts: `tools/lab/radiance/fn-helper*.sh`, `fn-profile*`,
`gpu-mailbox/`.
