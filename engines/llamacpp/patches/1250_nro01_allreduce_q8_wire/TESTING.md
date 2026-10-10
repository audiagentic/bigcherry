# 1250 historical testing — no current hardware campaign

Offline source-composition test: `PYTHONPATH=tools python -m unittest tools.tests.patch.test_1250_nro01_allreduce_q8_wire tools.tests.patch.test_1272_ar_host_compressed_wire` (requires a locally available pinned llama.cpp checkout). This checks 1252→1272→1250 patch composition and source assertions; **it does not qualify** P2P, graph replay, residual numerical equivalence or fallback atomicity.

**Withdrawn:** the former instructions to compare `GGML_CUDA_AR_P2P` on/off on dual gfx1100 and then toggle `GGML_CUDA_AR_FUSED_RESIDUAL=1`. The mandatory 1252 provider is rejected after a real hardware fault, so no 1250 hardware queue is permitted on the present no-P2P topology. Do not bypass dependency resolution.

PNRO02 next gate is **read-only**: record an actual production AR→mirrored residual ADD, provider and ADD exclusive wall time, then close if absent or <2.913% E2E time (insufficient for 3% theoretical speedup). If profitable, validate a separate default-off **exact-F32** host/CPU-root finish without 1252, including full-vocab/greedy/MTP parity, multi-request, graph replay, rank/alias checks and >=4-session paired ABBA hardware evidence. See PNRO02 / BCOP108.
