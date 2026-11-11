# rad0016_vtp14_rank_code_selection

**Status:** untested
**Plan item:** VTP14 (N1)

- Per-rank kernel code eligibility (`KernelRow::code_ranks`), with rank-aware `select`, `resolve` and `build_bands`.
- An explicit, ordered rank-to-card list, `--rank-devices`. It is the only way to get a mixed gfx1100+gfx1201 world.
- After declare, every rank gets the reference-kernel gate, a collective-agreement check and a receipt.
- Inert without the flag.
- The N3 cross-library rendezvous is not addressed here; see README.md.
