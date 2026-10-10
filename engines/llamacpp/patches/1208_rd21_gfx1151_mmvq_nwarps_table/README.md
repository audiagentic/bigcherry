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


## Static validation-package repair (2026-10-08)

The existing Experiment Contract is explicitly bound and `validation.toml` declares apply, build, correctness, activation, performance, and controls. This repairs the former `patch-lint` missing-binding/missing-adapter errors; it does **not** certify a validation campaign or introduce measurements. The declared activation marker does not yet exist in `patch.py`, so a new validation attempt must remain BLOCKED until real subject-hit/control-miss instrumentation and evidence are added. This patch remains untested/deferred on the unavailable gfx1151 hardware.
