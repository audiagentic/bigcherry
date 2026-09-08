# 1252_nro03_allreduce_p2p_provider

Plan `NRO03`; state `untested`; requires validated `1001_hip_internal_allreduce`.

The source fork adds optional two-GPU P2P copies. BigCherry has real gfx1100 evidence that destination-current/pull peer copies can return success with wrong data. Therefore this package is not a verbatim port. Its first draft encodes only two invariants: P2P is opt-in (`GGML_CUDA_AR_P2P`, default off), and every peer copy helper sets the **source** device current before issuing source->destination transfer.

No live collective calls the helper yet. Before wiring, the package requires bidirectional nonzero startup probes, per-direction streams/events, destination scratch ownership, and automatic fallback to host staging. Kernel direct peer reads/writes are out of scope.

A future performance campaign is valid only after every measured transfer is content-checked. API success/capability bits are not correctness evidence.
