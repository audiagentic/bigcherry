# libr2

## Summary

A subset of Radiance device kernels adapted for RDNA2, built against the
pinned engine sources. It currently supplies expert computation and
activation quantisation for the remote-MoE worker; it is not a complete
replacement for Radiance's upstream kernel library.

## Outcome

Limited to the rows listed in `units.txt`. Unsupported integer-matrix and
attention units are omitted. Each added row requires a successful build and
comparison against the host reference; the component has no standalone
self-test suite yet.

## Detail

The `r3_mma16.h` compatibility tile uses shared-memory permutation and
FP32 operations where RDNA2 lacks WMMA. `r2_registry.cpp` registers only
built rows; `rewrites.txt` holds the necessary source rewrite. The build
reuses the shared rewriter and compatibility headers from `../libr3`.

The [remote-MoE component](../moe-remote/README.md) is its main consumer.
See `docs/planning/active/patching-radiance-rdna3/RAD23.md` and RCS07
for design and qualification history; [component.toml](component.toml)
declares the build outputs.
