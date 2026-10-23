# 1252_nro03_allreduce_p2p_provider

**REJECTED / default off. Do not select or queue on the current dev dual-gfx1100 topology.** Hardware fault on 2026-10-07; b11474 non-composition and experiment removal recorded 2026-10-08. `patch.toml` and `SUMMARY.md` are authoritative for rejected status. PNRO03 owns any conditional requalification; PNRO18 is retired as duplicate probe hardening (BCOP292).

Port of nasone `7c5bb5cb` to the native HIP internal-AllReduce `ggml/src/ggml-cuda/allreduce.cu`. The opt-in `GGML_CUDA_AR_P2P=1` requires bidirectional peer capability/enable and a content-checked startup probe; on failure it retains host staging. Each peer copy is source-current on a source-owned stream and records directional events. `BIGCHERRY_PATCH_TRACE` can emit `BIGCHERRY_PATCH_HIT patch=1252_nro03 path=allreduce_p2p_source_push` only if the path executes.

**Evidence boundary:** `evidence/validation.json` has four failed-activation receipts, so requested-P2P throughput is not proof of actual P2P execution. The probe covers 4 KiB, 64 KiB+256, 1 MiB and 4 MiB (two passes/both directions) on temporary buffers, not production `dev_tmp`/event reuse or threshold-edge triples. These are coverage gaps, not an established fault cause. Existing `tools/tests/fixtures/hardware/p2p_copy_correctness_test.cpp` tests `hipMemcpyDefault`, not the async peer path.

Do not alter patch code or revive the removed experiment for this audit. PNRO03 documents the only possible new-evidence gate: changed physical/driver peer routing, byte-exact production-path test and actual activation, then all-rank correctness and matched performance. No current gain has been demonstrated.
