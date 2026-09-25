# Testing — 1250 (PNRO01 Q8 wire, PNRO02 fused residual)

Hardware: Brutus 2x gfx1100, `-sm tensor`, Qwen3.8-27B Q8_0 (MTP) and a dense control model.

Arms (single variable each, base = b11126 + 1252 with P2P off):
1. wire: unset vs `GGML_CUDA_AR_WIRE=q8_0` — lossy: full-vocab logprob comparison against unset, report max/mean divergence and greedy-token agreement; perf tg128/tg512/pp4096.
2. fusion: unset vs `GGML_CUDA_AR_FUSED_RESIDUAL=1` — must be exact (bit-identical greedy output, full-vocab diff at FP noise); perf as above plus ADD kernel count.
Each arm needs its BIGCHERRY_PATCH_HIT marker in the subject log and none in control. MTP acceptance must match within noise.
