# 1252_nro03_allreduce_p2p_provider

**Rejected at b11474 (2026-10-08); archived, not supported for execution.** PNRO03 and PNRO18 are closed. Keep `patch.toml state="rejected"`; do not enable, promote or queue the package on the existing no-P2P topology.

Historical implementation: port of nasone `7c5bb5cb`, with source-current push, two directional streams/events, four-size startup byte probe, `GGML_CUDA_AR_P2P=1` opt-in and a `BIGCHERRY_PATCH_HIT patch=1252_nro03` marker. The probe uses fresh buffers and synchronous verification; production uses `p->dev_tmp` and asynchronous cross-device event slots. This is a coverage gap, **not a proven fault cause**.

The stored `evidence/validation.json` reports full-vocabulary bit identity **with failed activation** and `contract passed=false`: fallback equivalence, not P2P correctness or speed. Native HIP internal AllReduce is upstream (#27825). See `SUMMARY.md`, `TESTING.md`, PNRO03, PNRO18 and BCOP102.
