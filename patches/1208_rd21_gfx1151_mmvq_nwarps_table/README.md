# 1208 RD21 gfx1151 MMVQ table

This package owns the RD21 implementation and its activation-gate evidence.
The evidence records that the patch is blocked on RDNA3.5/gfx1151 hardware;
it is not a performance validation result.

**Parked (2026-09-13, explicit user instruction):** all gfx1151/RDNA3.5
work is deliberately parked for this project pending real gfx1151
hardware access. `deferred-hardware` remains the correct tracked
disposition (not `rejected`/`superseded` -- this project's own lifecycle
rules reserve those for a failed/replaced candidate, never for
temporarily-unavailable hardware). No further validation work on this
patch is expected until gfx1151 hardware becomes available; do not
infer coverage from gfx1100/gfx1201/gfx1030 results.
