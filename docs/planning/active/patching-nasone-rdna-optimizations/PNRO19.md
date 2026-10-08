---
id: PNRO19
order: 0
plan: patching-nasone-rdna-optimizations
state: completed
created-at: '2026-09-27T12:55:18.984027+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: S
---

# Patch 1270 anchor failures — already fixed

## Disposition (2026-10-08)

**Completed / duplicate of implemented fix.** The historical `test_apply_and_idempotent` failures were not an unresolved vendor-source drift:
- `pnro14-includes`: `max_span_lines=2` rejected an anchor with two newline characters (three spanned lines). Fixed to 3.
- `pnro14-device-select`: literal comment text in the anchor did not match the patcher's noise-stripped C source. Fixed by using `csource.strip_noise(_DEVICE_OLD, "c")`.

Commit `f4228540b774afee0efb2f8e31e760b0a3c65f67` (2026-09-27) implemented both changes in `patches/1270_pnro14_rdna35_fa_tile_d256/patch.py`. Current `tools/tests/patch/test_1270_pnro14_rdna35_fa_tile_d256.py` retains apply/idempotence and missing-host-selector fail-closed tests. No new patch-author task or duplicate plan is needed. The 2026-10-08 audit verified source assertions only; it did not rerun the patch suite.

**Remaining work belongs to PNRO14**, not PNRO19: gfx1151 target selection, hardware correctness, and performance are unqualified. PNRO14's terminal no-qualified-lane disposition supersedes immediate rework of 1270.

## History

Created 2026-09-27 after pre-existing offline test failures; fixed later the same day in `f422854`. No independent changes to this capability in the preceding 12 hours.
