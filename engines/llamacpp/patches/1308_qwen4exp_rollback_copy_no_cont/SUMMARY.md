# 1308_qwen4exp_rollback_copy_no_cont

**Status:** validated
**Plan item:** QFP13

## What it does

With `BIGCHERRY_ROLLBACK_NO_CONT=1`, the Qwen4Exp recurrent conv-state builder writes each rollback snapshot
(n_rs_seq + 1 slots per recurrent layer, 4 under MTP3) as `ggml_cpy(tail, dst)` instead of
`ggml_cpy(ggml_cont(tail), dst)`: one copy kernel per slot instead of two. The decode kernel census attributes ~89
of ~108 runtime copy launches per generated token to this sequence, so this removes ~45 launches per token per
GPU (decode is launch-gap bound, QFP13). The copy is exact: outputs must be bit-identical.

## Hardware result (2026-10-04 review)

Profile v3 adoption ABBA (with 1307/1309/1310): +3% ~10K, +3.6% ~80K, greedy identical; kernels/token 1270 -> 1212 on its own.

## 2026-10-14 b11552 source-route qualification (BCOP209)

1308 remains validated/default-on. The Qwen4Exp rollback `VIEW_3D` source and flattened `VIEW_2D` cache destination have different shapes for multi-channel state. Pinned `cpy.cu::ggml_cuda_cpy_as_memcpy_2d` therefore rejects the direct-strided route; F32-to-F32 uses the scalar copy kernel rather than a guaranteed one-shot DMA copy. The eliminated CONT is a graph materialization; the final state write is mandatory. Historical ~89/108 runtime copy launches cannot be reclassified by subtracting launch counts because kernel families can change. See QFP13 for the residual-copy census and correctness/acceptance gate. No new b11552 hardware measurement or patch change is claimed.
