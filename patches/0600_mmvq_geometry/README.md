# 0600_mmvq_geometry framework validation

Adds two defaulted template parameters (`nwarps_explicit`,
`rows_per_block_explicit`) to the MMVQ kernel template. MMVQ derives its
geometry from `calc_nwarps`/`calc_rows_per_block` at compile time, so
(unlike MMQ/MMVF/MMF) an alternative geometry needs genuinely new compiled
code rather than a runtime argument. Zero means "derive as upstream does",
so every pre-existing instantiation is unchanged, including its launch
bounds; only generated variants name both parameters. Geometry bounds are
`static_assert`ed inside the kernel as a backstop against a generator bug.

This local framework adapter has no Experiment Contract: it checks build
plumbing, not a claimed kernel speedup. Historical `validated` state is not
current qualification.

Validation is the universal `apply`/`build` checks only -- no patch-specific
custom check exists for this package. Passing proves the patch applies
cleanly to the pinned upstream source and the resulting tree compiles (i.e.
the new template still instantiates the native/default case correctly at
compile time); it does NOT prove any generated non-native geometry actually
runs correctly or with the expected performance.

## Upstream / provenance

Local design, part of this project's own HIP measured-dispatch framework
(HI09 part 1).
