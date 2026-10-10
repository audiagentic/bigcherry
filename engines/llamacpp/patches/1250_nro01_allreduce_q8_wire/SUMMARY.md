# 1250_nro01_allreduce_q8_wire

**Status:** evaluated (not validated; not promotable on current no-P2P hardware)
**Plan items:** PNRO01 / PNRO02

1250 depends on rejected 1252 P2P plus untested 1272 wire. Its residual hook/matcher exists, but the fused API admits only **two-rank Q8_0** with a positive copy threshold; it cannot implement PNRO02's proposed exact-F32 fusion. The existing 1250 mechanics tests verify source composition/idempotence, not graph capture, safe fallback after partial enqueue, full-vocabulary parity, or measured improvement. Presence-based `GGML_CUDA_AR_FUSED_RESIDUAL` parsing also treats `0` as enabled.

**Terminal package decision:** Do not re-enable or benchmark 1250 on current topology. Keep historical package state/evidence unchanged; future residual work belongs to PNRO02 and an eligible exact-F32 production finish (PGC09/PGC12/1291), with 1272 retaining Q8 codec ownership. Do not create another transport/cache/scheduler. BPB01 already records 1250 as superseded/infeasible; BCOP108 records the disposition.
