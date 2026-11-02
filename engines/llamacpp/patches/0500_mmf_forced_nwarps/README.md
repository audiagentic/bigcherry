# 0500_mmf_forced_nwarps framework validation

## Summary

Same explicit-appended-defaulted-parameter shape as HI07, applied to MMF's three dispatchers (which share an identical signature/call tail); shared-memory sizes are recomputed from the forced nwarps immediately after the scan so allocation stays correct.

## Outcome

Recorded state: `validated` ([patch.toml](patch.toml)); qualification and limitations: [SUMMARY.md](SUMMARY.md).

## Detail

Same shape as 0400 (HI07): a forced MMF nwarps value travels as an explicit,
appended, defaulted parameter through MMF's three dispatchers, which share
an identical signature/call tail, so the native path stays byte-identical to
upstream. Dynamic shared-memory sizes (`nbytes_shared_iter`,
`nbytes_shared_combine`) are recalculated from the forced value immediately
after the scan -- applying it later would require hand-recomputing them and
would under-allocate for a forced nwarps larger than native's choice.

This local framework adapter has no Experiment Contract: it checks build
plumbing, not a claimed kernel speedup. Historical `validated` state is not
current qualification.

Validation is the universal `apply`/`build` checks only -- no patch-specific
custom check exists for this package. Passing proves the patch applies
cleanly to the pinned upstream source and the resulting tree compiles; it
does NOT prove runtime activation, dispatch correctness, or performance for
this patch's actual behavior.

## Upstream / provenance

Local design, part of this project's own HIP measured-dispatch framework
(HI08).
