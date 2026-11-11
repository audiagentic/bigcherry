# rad0016_vtp14_rank_code_selection

**Status:** untested
**Plan item:** VTP14 (N1)

Scope: per-rank kernel code eligibility for a mixed gfx1100+gfx1201 tensor-parallel world. Inert without
`--rank-devices`. The ROCm lane build and the HIP unit tests pass. On cards, per-rank selection works, and serving
stops at the N3 all-reduce rendezvous.

## Summary

Lets each tensor-parallel rank select the kernels its own card runs, so XTX (libr3) and R9700 (libr4d) ranks can share one world when named explicitly with `--rank-devices`.

- Per-rank kernel code masks (`KernelRow::code_ranks`), with rank-aware `select`, `resolve` and `build_bands`.
- An explicit, ordered rank-to-card list, `--rank-devices`. It is the only way to get a mixed-architecture world.
- After declare, every rank gets the reference-kernel gate, a collective-agreement check and a receipt.

## Outcome

Experimental and not promoted. Any mixed-library world (libr3 + libr4d), TP2 included, hangs in all-reduce init
until the N3 shared rendezvous lands.

## Detail

See README.md and docs/planning/active/run-dense-27b-virtual-tp/VTP14.md.
