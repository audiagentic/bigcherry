# 1239_hi119_fused_moe_glu_test_case

Plan item HI119 (superseded by THA02, `tuning-hip-autotune`). See
`SUMMARY.md` for the mechanism.

## Status

Same schema-2 gap as `1238`: this patch's fused-GLU test case was built and
briefly exercised on real 4-GPU hardware during HI119's original session,
but that evidence is for a pre-schema-2 build and was deliberately not
carried forward as validating STATE. dev-gpt-agent's design review (two
rounds) also required a real-execution proof this patch's own harness must
supply -- that the fused signature was *actually* dispatched as fused
(observed signature digest == requested, observed candidate resolution ==
expected), not merely that CPU-vs-GPU numerics matched -- since a future
CUDA fusion-detection regression could otherwise silently fall back to
unfused execution while still passing numeric comparison. That check is
part of THA02's scope, not yet re-verified against a schema-2 build.

Depends on patch `1238`'s deterministic routing.

## Real schema-2 hardware pass (2026-09-12, Brutus dual gfx1100, THA02)

All 4 registered instances of this patch's `test_bigcherry_moe_glu_fusion`
class (SWIGLU/GEGLU x the real production broadcast Q8_0 k=2048 shape x a
non-broadcast F32 k=256 shape) passed cleanly and reproducibly on both real
XTX GPUs (ROCm0 and ROCm1, individually, 136/136 each run), real
correctness errors 1e-8 to 1e-14 against the CPU reference, far under the
5e-3 threshold. Full detail in THA02's plan item, including a separate
gfx1201/gfx1030 finding (THA33) confirmed unrelated to this patch.

The dispatch-execution-proof requirement (observed signature digest ==
requested, proving the fused path actually executed rather than silently
falling back to unfused) was not separately re-verified this pass -- still
open before promotion.

## Disposition

`state` stays `"untested"`. `kind = "diagnostic"`. Hardware-confirmed on
gfx1100 as of 2026-09-12; the dispatch-execution-proof requirement is still
open before any promotion.
