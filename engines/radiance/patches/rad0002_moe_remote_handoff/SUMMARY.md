# rad0002_moe_remote_handoff

**Status:** validated
**Plan item:** RAD01

## Summary

Engine-side hooks (14 call sites, 4 files) for the whole-expert MoE remote hand-off.

## Outcome

Validated for the Radiance v1.14.0 production `serving-core` patch set on
3 November 2026. Qualification is profile-scoped to Qwen3.8-Flash-Next
fp8-iq4r-moe on an R9700 engine with RX 7900 XTX expert helpers. The
recorded quality, repeatability and helper-activation checks passed; this
does not establish equivalent behaviour on other models or layouts.
See the [promotion evidence](evidence/promotion.json) and
[qualification record](README.md).


## Detail

See below.

## What it does

Adds the engine side of the whole-expert MoE hand-off as thin hooks into one BigCherry-owned extension point (`abi/bc_moe_remote.h`, `core/place/bc_moe_remote.cpp`, `core/place/bc_moe_remote_place.h`, `arch/common/bc_moe_remote_block.h`, all created by this package). When the libr3moe plugin has registered and the row policy admits a step, the routes the helper owns are published to it and masked out of the local grouped GEMMs; the helper rows then rejoin the block where the gather would read them. Where the write behind the block takes the gather (a tensor-parallel engine's fused MoE gather + reducing hc write), `moe_mailbox_inject` gives each helper row a spare sorted row of that gather with weight 1, on the layer's consumer rank only, so the reduction sums each helper row once and no standalone all-reduce runs (RAD49). Such a step is published with `fused = 1`, so it never asks for the helper combine (`R3_MOE_WORKER_COMBINE`), which only the add reads; `RADIANCE_DEBUG_ROUTING` prints the layer's router weights before the inject sets the injected slots' weights to 1. Otherwise (a one-rank engine, a separate FP32 combine, or rows the fused write does not take) `moe_mailbox_add` sums the local and helper rows in slot order in place of `moe_gather`, and the engine's own write (reducing or not) reads that output. The engine's all-reduce path is upstream's in both cases. With no plugin or an excluded row count nothing changes.

The upstream files receive 11 one-line guarded call sites (the earlier port was 20 edits and about 170 core lines):

| File | Sites |
|---|---|
| `arch/common/rad_block_moe_fp8.h` | include; `BcMoeRemote bc_remote` member; `declare` (publishes the layer map, declares the three mailbox ops from the plugin's rows); `after_route` (publish and re-scatter); `merge` (injects into the fused gather, or replaces the gather) |
| `core/place/planner.cpp` | include; `pin_mapped` (`bc_moe_remote_pin_unit`); `weight_footprint` (`bc_moe_remote_mark_footprint`); `rank_units` (skip preplaced units, the one policy line that cannot leave the core) |
| `core/engine_bringup.cpp` | include; `bc_moe_remote_publish_placeholder` before the tier check |

All policy is in the created files and the plugin: the row gate (`R3_MOE_REMOTE_MIN_ROWS` / `MAX_ROWS`), the plugin registry (`bc_moe_remote_register`, replacing the two `dlsym` hooks; layer maps published before registration are replayed), the layer map, the `R3_MOE_WORKER_SKIP_OWNED` placement rule, the never-read mapped placeholders, and the choice between inject and add (`BcMoeRemote::merge`, on the same `gather_taken` the block and HyperConn use). The shadow compare and the mask-only mode are not here: they are `rad0004_moe_remote_diagnostics`, for a diagnostic build only. A production engine that finds `R3_MOE_WORKER_SHADOW` or `R3_MOE_WORKER_MASK_ONLY` set answers "no remote rows" and says so once.

Changes of behaviour against the 1.3.0 port: the step gate is decided on the step's row count `T` everywhere (identical to the pass rows for decode-sized single-pass steps), the all-reduce is no longer forced (RAD49: the helper rows join the fused gather instead), and a missing mailbox op while the gate is open is a declare error.

## Why

The seam allows a helper GPU to compute cold routed experts rather than making the serving ranks repeatedly consume host-resident expert weights across the link.

## Evidence

The [v1.14.0 promotion record](README.md) gives the matched quality
checks, helper activity, captures and limits. The
[recorded promotion evidence](evidence/promotion.json) identifies the
qualified package. The older 1.3.0 measurements and 1.7.0 port history
are historical context, not current-pin performance evidence; see
[the Radiance archive](../../../../docs/archive/radiance/).


## Upstream

Local BigCherry integration patch, now validated against the pinned
Radiance v1.14.0 source. The original implementation was developed
against Radiance 1.3.0 and reworked for 1.7.0 before this promotion.
The production patch is selected by `serving-core` in
[`engines/radiance/recipes.toml`](../../recipes.toml).
