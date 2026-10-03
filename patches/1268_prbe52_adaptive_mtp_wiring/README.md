# 1268 PRBE52 adaptive MTP wiring

Untested runtime wiring for the existing 1255 adaptive MTP controller. The default remains fixed-depth; `n_min_adaptive=0` is disabled.

## Composition

`common/common.h` anchors only on the pinned b11126 `n_min` declaration so lower-order composed fields adjacent to `n_max` do not invalidate the patch. Runtime wiring requires `1255_nro06_adaptive_mtp_depth`.

## Validation

Contract `PRBE52-ADAPTIVE-MTP-WIRING`; mechanics require apply/idempotence and anchor-fail-closed behavior. Hardware qualification uses greedy-token parity plus batched MTP throughput on gfx1100/gfx1201. State remains `untested` until fresh evidence is collected.
