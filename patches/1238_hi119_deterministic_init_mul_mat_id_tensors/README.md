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

A fresh schema-2 Brutus 4-GPU correctness + record-mode structural-match
validation, plus a real HI83-format evidence record, is the explicit
required next step -- tracked under THA02
(`docs/planning/active/tuning-hip-autotune/THA02.md`), not yet run.

## Disposition

`state` stays `"untested"`. `kind = "diagnostic"` (test-harness patch, not
a production dispatch patch). Do not promote or re-validate against stale
schema-1 evidence; THA02 owns the fresh schema-2 hardware pass this needs.
