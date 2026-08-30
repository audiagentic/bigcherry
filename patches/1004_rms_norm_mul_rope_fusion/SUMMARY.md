# 1004_rms_norm_mul_rope_fusion: Upstream backport: fuse rms_norm + mul + rope (+ view + set_rows)

**Status:** rejected
**Group:** upstream-fixes
**Plan item:** none

## What it does

Was intended to fuse the per-layer rms_norm/mul(scale)/rope(+optional view/set_rows) chain into a single kernel launch instead of upstream's four-to-five separate launches, entirely outside the MUL_MAT-family dispatch-engine work.

## Why

REJECTED: the full fusion (kernel, should_fuse, dispatch wiring) already existed in this project's own pinned base before this patch was ever ported -- all 5 edits verified already-applied at both the b10362 and b10502 pins. Silent no-op since porting, not a functional change; left in the tree for provenance with STATE=rejected.

## Upstream / provenance

Cherry-picked from merged upstream commit 687e7789271ec1276e3470f158428e11a4f80b6f (PR #26767). Turned out to already be ancestral to this project's pinned base.
