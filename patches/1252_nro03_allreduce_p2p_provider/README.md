# 1252_nro03_allreduce_p2p_provider

Plan `NRO03`; state `untested`; requires validated `1001_hip_internal_allreduce`.

The source fork adds optional two-GPU P2P copies. BigCherry has real gfx1100 evidence that destination-current/pull peer copies can return success with wrong data. Therefore this package is not a verbatim port. Its first draft encodes only two invariants: P2P is opt-in (`GGML_CUDA_AR_P2P`, default off), and every peer copy helper sets the **source** device current before issuing source->destination transfer.

No live collective calls the helper yet. Before wiring, the package requires bidirectional nonzero startup probes, per-direction streams/events, destination scratch ownership, and automatic fallback to host staging. Kernel direct peer reads/writes are out of scope.

A future performance campaign is valid only after every measured transfer is content-checked. API success/capability bits are not correctness evidence.

## Real hardware evidence (2026-09-12, Brutus, gfx1100)

Found by inspection (not yet built): the same compile-breaking anchor bug
found and fixed in patch 1250 (NRO01) this same session -- the
`p2p-request-init` edit's anchor ended at the `=` sign of a single-line
statement, which `insert_after` would have corrupted identically. Fixed
proactively using the same LITERAL-placeholder technique before spending a
real build cycle to rediscover the same bug. Re-verified with a real
isolated build (`1001_hip_internal_allreduce` + this patch, gfx1100,
`bigcherry-native` composition): clean build, `BUILD_OK`. This closes this
draft's own "applies cleanly, builds HIP" bar for the first time.

No other acceptance criteria for this draft are addressed by this fix --
the P2P helper remains unwired (no live collective calls it), and the
bidirectional-probe/scratch-ownership/fallback work this README's own
"Before wiring" paragraph describes is entirely separate, not-yet-started
work.
