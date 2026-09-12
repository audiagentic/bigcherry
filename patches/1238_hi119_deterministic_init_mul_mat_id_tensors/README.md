# 1238_hi119_deterministic_init_mul_mat_id_tensors

Plan item HI119 (superseded by THA02, `tuning-hip-autotune`). See
`SUMMARY.md` for the mechanism.

## Status

This patch's own mechanism (seeded shuffle of `init_mul_mat_id_tensors()`'s
full `[0, n_mats)` range) was implemented and briefly validated on real
4-GPU hardware during HI119's original session, but HI119's own review
found a real ABI gap afterward: the dispatch signature schema was bumped
(`GGML_HIP_SIGNATURE_SCHEMA_VERSION` 1->2) to resolve a legacy-provenance
ambiguity, which means the binary that earlier real-hardware pass validated
no longer matches what a fresh build produces. HI119's STATE was
deliberately flipped back to `untested` for exactly this reason -- not a
regression, a genuine "the evidence is for a build that no longer exists"
situation.

## Real schema-2 hardware pass (2026-09-12, Brutus dual gfx1100, THA02)

Built the full chain [1222,1223,1236,1238,1239,1240] at current pin b10901
(schema v2). Real correctness net on ROCm0: 2162/2162 passed (deterministic
seed=42). Tightly filtered to the 4 registered fused-GLU instances (which
depend on this patch's deterministic routing): 136/136 passed reproducibly
on ROCm0 and ROCm1 individually (the real dual-XTX production topology).
This satisfies the schema-2 hardware requirement below for gfx1100.

A separate real finding surfaced on gfx1201/gfx1030 (not this patch chain's
target architecture) -- investigated at length and confirmed unrelated to
this patch (crash stack is entirely in pre-existing production
`mul_mat_vec_f_cuda`, never touching test-harness code this patch changes).
Filed as `THA33`. Does not block this patch's gfx1100 confirmation.

A real HI83-format evidence record is still a separate, not-yet-done step
before any promotion.

## Disposition

`state` stays `"untested"`. `kind = "diagnostic"` (test-harness patch, not
a production dispatch patch). Hardware-confirmed on gfx1100 as of
2026-09-12; do not promote without a real HI83-format evidence record.
