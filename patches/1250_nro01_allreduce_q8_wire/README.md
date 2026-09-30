# 1250_nro01_allreduce_q8_wire

Plans `PNRO01` (P2P Q8 wire) and `PNRO02` (fused residual). Requires both `1252_nro03_allreduce_p2p_provider` and `1272_ar_host_compressed_wire`.

Ownership after the b11233 migration is deliberately split: 1272 owns `GGML_CUDA_AR_WIRE`, the single Q8_0 block-32/fp16-scale codec, host/mapped transport, and shared finish kernels. 1250 owns only the provider extension that routes 1272 Q8 copy-engine traffic through 1252 P2P when available plus the optional meta-backend residual-ADD fusion. There is no compatibility copy of the old 1250 codec/parser/finish implementation.

Switches: `GGML_CUDA_AR_WIRE=q8_0` (parsed by 1272) selects Q8; `GGML_CUDA_AR_FUSED_RESIDUAL=1` enables PNRO02. Unset wire keeps 1272/pristine selection semantics and 1250 does not fire.

Activation markers (`BIGCHERRY_PATCH_TRACE`): `patch=1250_nro01 path=allreduce_q8_0_p2p_shared` when Q8 uses the P2P provider; `patch=1250_nro02 path=allreduce_fused_residual` only after fused execution succeeds. 1272 independently emits `patch=1272_ar_wire path=ar_wire_q8_0` for explicit Q8 selection.

Provenance: nasone commit `e06dcf6300718227cb8cfda9e61fb12ccb693418`; provider ordering remains the source-current 1252 enqueue loop, with codec/fused-finish ownership migrated to 1272 under the no-legacy policy.
