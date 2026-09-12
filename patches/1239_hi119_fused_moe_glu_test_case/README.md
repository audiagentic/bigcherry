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

## Disposition

`state` stays `"untested"`. `kind = "diagnostic"`. THA02
(`docs/planning/active/tuning-hip-autotune/THA02.md`) owns the fresh
schema-2 hardware pass and the dispatch-execution-proof requirement before
any promotion.
