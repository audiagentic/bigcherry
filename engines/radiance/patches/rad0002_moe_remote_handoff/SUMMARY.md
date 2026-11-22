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

Adds the engine side of the whole-expert MoE hand-off as thin hooks into one BigCherry-owned extension point (`abi/bc_moe_remote.h`, `core/place/bc_moe_remote.cpp`, `core/place/bc_moe_remote_place.h`, `arch/common/bc_moe_remote_block.h`, all created by this package). When the engine has opened its remote-MoE generation (the libr3moe factory, resolved after the engine claims its slots) and the row policy admits a step, the routes the helper owns are published to it and masked out of the local grouped GEMMs; the helper rows then rejoin the block where the gather would read them. Where the write behind the block takes the gather (a tensor-parallel engine's fused MoE gather + reducing hc write) and the step is at most one sequence's decode or verify step (`1 + max_spec` rows, where the fused form measured faster), `moe_mailbox_inject` gives each helper row a spare sorted row of that gather with weight 1, on the layer's consumer rank only, so the reduction sums each helper row once and no standalone all-reduce runs (RAD49). Such a step is published with `fused = 1`, so it never asks for the helper combine (`R3_MOE_WORKER_COMBINE`), which only the add reads; `RADIANCE_DEBUG_ROUTING` prints the layer's router weights before the inject sets the injected slots' weights to 1. Otherwise (a one-rank engine, a separate FP32 combine, a larger step, or rows the fused write does not take) `moe_mailbox_add` sums the local and helper rows in slot order in place of `moe_gather`; behind a HyperConn write the block then runs its own all-reduce and the write is the plain one (the v25 path, `issues_ar` / `takes_ar` through `bc_moe_remote_hc_open`), and a norm-fused consumer keeps its fold. With no plugin or an excluded row count nothing changes.

The upstream files receive 14 one-line guarded call sites (the earlier port was 20 edits and about 170 core lines):

| File | Sites |
|---|---|
| `arch/common/rad_block_moe_fp8.h` | include; `BcMoeRemote bc_remote` member; `declare` (publishes the layer map, declares the three mailbox ops from the plugin's rows); `after_route` (publish and re-scatter); `merge` (injects into the fused gather, or replaces the gather); `issues_ar()` (the block's own all-reduce on a handed-off step that does not fuse, behind a HyperConn write) |
| `arch/common/rad_block_hc.h` | include; `takes_ar()` |
| `core/place/planner.cpp` | include; `pin_mapped` (`bc_moe_remote_pin_unit`); `weight_footprint` (`bc_moe_remote_mark_footprint`); `rank_units` (skip preplaced units, the one policy line that cannot leave the core) |
| `core/engine_bringup.cpp` | include; `bc_moe_remote_publish_placeholder` before the tier check |

All policy is in the created files and the plugin: the frozen row policy (the `remote_moe` settings of the serving manifest, read once when the generation opens; no environment gates), the engine-owned generation that replaces the process-global plugin registry (`bc_moe_remote_open_with` through the plugin's factory; no registration call and no layer-map replay, the layer map is per generation), the `skip_owned` placement rule, the never-read mapped placeholders, and the choice between inject and add (`BcMoeRemote::merge`, on the same `gather_taken` the block and HyperConn use). The shadow compare and the mask-only mode are not here: they are `rad0004_moe_remote_diagnostics`, for a diagnostic build only. A production engine whose manifest asks for the shadow or mask-only mode answers "no remote rows" and says so once.

Changes of behaviour against the 1.3.0 port: the step gate is decided on the step's row count `T` everywhere (identical to the pass rows for decode-sized single-pass steps), the all-reduce is forced in `issues_ar()` only on handed-off steps that do not fuse (RAD49), so the pipelined prefill callers agree with `pass()`, and a missing mailbox op while the gate is open is a declare error.

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
