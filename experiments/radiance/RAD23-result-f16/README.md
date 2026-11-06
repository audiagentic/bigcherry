# RAD23: fp16 result rows with power-of-two row exponent (protocol v21)

Archived from `feat/rad23-result-f16` (tip `c973b18e4`, 2026-11-01). The inert `change.patch` is the single v21 code commit `1f77761e8`, based on `3732ffc59` (the archived v20 f32 variant); the subsequent RAD39 planning edit `c973b18e4` is already on main and is excluded from the patch.

## What and why

Compress helper result rows to fp16 with one power-of-two row exponent so fp32 partial range can be retained more accurately than ordinary fp16 result planes. It changes result-plane readers/writers, mailbox protocol, combine kernels and tests.

## Outcome and why stopped

**(a): RAD39 KLD planning note already on main; (b): v20 prerequisite subsequently superseded by production protocols; (c): one novel v21 row-exponent implementation preserved in the patch.** Current main uses later helper-combine protocols (including #497, v22), not this v21 format. The branch does not record an end-to-end KLD or prefill/decode A/B for its fp16+exponent variant; the specific stopping decision is **unknown**, not a measured failure.

## Evidence and resumption

- `1f77761e8` (v21 implementation), `c973b18e4` (RAD39 note); `docs/planning/active/patching-radiance-rdna3/{RAD23,RAD39}.md`; PR #497.
- Recreate the historical v20 base by applying `experiments/radiance/RAD23-result-f32/change.patch` to `a20d1cb28`, then `git apply --3way experiments/radiance/RAD23-result-f16/change.patch`. On current main, redesign protocol integration instead of patching directly; run component, KLD and performance gates.
