# 1250_nro01_allreduce_q8_wire — historical evaluated package

**State:** `evaluated` (unchanged); **not eligible for deployment/qualification on current BigCherry topology**. Plans PNRO01/PNRO02. Package `requires` both `1252_nro03_allreduce_p2p_provider` (**rejected** following the 2026-10-07 hardware fault) and `1272_ar_host_compressed_wire` (**untested**). Do not override the dependency check to enable residual fusion.

Source implementation: 1250 adds a Meta backend reduction→reshape→mirrored-F32-ADD matcher, optional `ggml_backend_comm_allreduce_tensor_fused_add_t` hook, and suppression of the ADD after a successful fused finish. It reuses 1272's Q8_0 codec and finish kernels and 1252's P2P provider, with a host-copy fallback inside the composed implementation. **No exact-F32 fused residual path exists**: `ggml_cuda_ar_allreduce_fused_add` accepts only explicit `GGML_CUDA_AR_WIRE=q8_0`, two devices and a positive copy threshold. Q8 is lossy; do not describe this as exact-F32 or bit-identical.

`GGML_CUDA_AR_FUSED_RESIDUAL=0` is treated as enabled by the current `getenv(...) != nullptr` check, if all other conditions hold. The trace `patch=1250_nro02 path=allreduce_fused_residual` is not proof of graph replay, numerical equivalence, or provider completion. No qualified PNRO02 hardware benefit is recorded.

**Disposition:** PNRO02 owns any future default-off **exact-F32** host/CPU-root epilogue, only after a read-only production graph census and >=3% optimistic E2E bound. Preserve PNRO03/1252 rejection and validated PGC09/PGC12/1291 production paths. See PNRO02 and BCOP108; do not queue the old Brutus P2P instructions.
