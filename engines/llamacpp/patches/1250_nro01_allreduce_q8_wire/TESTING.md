# Testing — 1250 (PNRO01 P2P Q8 wire, PNRO02 fused residual)

Offline mechanics: `PYTHONPATH=tools python -m unittest tools.tests.patch.test_1250_nro01_allreduce_q8_wire tools.tests.patch.test_1272_ar_host_compressed_wire`. The composition test applies pristine b11233 -> 1252 -> 1272 -> 1250, checks idempotence, checks a single codec/parser definition, and proves the P2P provider calls 1272's shared finish.

Hardware: Brutus 2x gfx1100, `-sm tensor`, Qwen3.8-27B Q8_0 (MTP) plus a dense control. For PNRO01 compare the same 1272 Q8 binary with P2P provider enabled/disabled; require `patch=1272_ar_wire path=ar_wire_q8_0`, and require `patch=1250_nro01 path=allreduce_q8_0_p2p_shared` only in the P2P arm. Report greedy agreement/full-vocab divergence and decode/prefill throughput. For PNRO02 toggle only `GGML_CUDA_AR_FUSED_RESIDUAL=1`; require `patch=1250_nro02 path=allreduce_fused_residual` only in the subject and compare output plus ADD-kernel count.
