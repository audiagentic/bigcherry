# rad0004_moe_remote_diagnostics

**Status:** untested
**Plan item:** MEN12

## What it does

Adds the remote-MoE diagnostics to the MoE block, for a diagnostic build only. Requires `rad0002_moe_remote_handoff`, whose extension point it hangs off.

- Shadow compare (`R3_MOE_WORKER_SHADOW`): the helper computes the routes it owns and compares them with the local result, which still goes forward unchanged; libr3moe reports the per-layer error at shutdown.
- Mask-only (`R3_MOE_WORKER_MASK_ONLY`): the local experts run unmasked, their rows are captured, the routes are published and masked late, local rows are restored to engine order, and the helper's rows (1: the engine's captured rows, 2: the helper's per-token rows) are merged.

The package creates `arch/common/bc_moe_remote_diag.h` (it `#error`s without `BC_MOE_REMOTE_DIAG`) and adds four call sites to `rad_block_moe_fp8.h` (include, member, declare, after-down), each wrapped in `#ifdef BC_MOE_REMOTE_DIAG`. Without the macro nothing of it is compiled, and a production engine that finds either variable set answers "no remote rows" (every expert computes locally) and says so once.

## Build and recipes

The macro is defined by `build.moe-diag` (`CMAKE_CXX_FLAGS=-DBC_MOE_REMOTE_DIAG`, which reaches the engine core and the arch plugins); the source is `radiance-diag` (the production patch sets plus `moe-remote-diag` = rad0002 + rad0004). `patch.toml` declares `diagnostic-build = "BC_MOE_REMOTE_DIAG"`, and patch-lint fails if this package enters any source, patch set or experiment whose name does not end `-diag`.

## Why

The diagnostics were part of rad0002 on 1.3.0 (about 70 lines of core code). Production hand-off needs none of them, and the owner's rule is that rad0002 carries thin hooks only; the compare and the late-mask path are kept for investigating a helper/engine disagreement.

## Evidence

Ported from the 1.3.0-era rad0002 shadow and mask-only edits, which were used to localise helper/engine disagreement (Flash-Next layers with protected experts; helper vs engine rel_l2 0.0011 at acceptance, RAD09). Not re-run on 1.7.0: no build or GPU run has exercised this port, so the package is `untested`. Only text-level application against the 1.7.0 fixtures is checked.

## Upstream

Local BigCherry change against Radiance 1.7.0 (pin `48f7b8eb6d703cd5318346b5a351a292afaf4bf9`, MEN12). No upstream equivalent.
