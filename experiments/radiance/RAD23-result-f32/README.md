# RAD23: unrounded fp32 helper result planes (protocol v20)

Archived from `feat/rad23-result-f32` (tip `3732ffc59`, 2026-11-01). Patch base: `a20d1cb28` (the prior RAD23 route-bounded quantise change, #484). The inert `change.patch` is a **code-only** diff from that base to the tip, excluding already-shipped planning and release-ledger history.

## What and why

An alternative two-to-four-helper row-combine design: write unrounded f32 expert partials into result planes and reduce after collection rather than relying on intermediate fp16 rounding. The branch includes pool-worker startup fixes and a cold-memory CPU combine bench.

## Outcome and why stopped

**(a): existing helper-count, base combine and gated quantise foundations on main (#404, #484); (c): four cohesive v20/result-f32 and CPU-benchmark commits not carried to main in this form.** Main subsequently chose chunk-wise helper combine and newer protocol revisions (e.g. #497, protocol v22), rather than this v20 result-plane format. The branch has no documented end-to-end KLD, quality or throughput A/B establishing acceptance. Why no further v20 validation was done is **unknown**; the design is now incompatible with the current wire layout.

## Evidence and resumption

- Branch commits `ed113386b`, `d800a459c`, `fb8af4bae`, `3732ffc59`.
- `docs/planning/active/patching-radiance-rdna3/RAD23.md`; `docs/planning/active/patching-radiance-rdna3/RAD39.md` (KLD noise-band gate); PRs #404, #484 and #497.
- Restore **on historical base** `a20d1cb28` via `git apply --3way experiments/radiance/RAD23-result-f32/change.patch`. Do **not** apply blindly to current main: reconcile result-plane ownership, protocol version and tests first, then qualify KLD and CPU/GPU performance. No GPU work was performed for this archive.
