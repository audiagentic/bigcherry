# 1308_qwen4exp_rollback_copy_no_cont

**Status:** evaluated
**Plan item:** QFP13

## What it does

With `BIGCHERRY_ROLLBACK_NO_CONT=1`, the Qwen4Exp recurrent conv-state builder writes each rollback snapshot
(n_rs_seq + 1 slots per recurrent layer, 4 under MTP3) as `ggml_cpy(tail, dst)` instead of
`ggml_cpy(ggml_cont(tail), dst)`: one copy kernel per slot instead of two. The decode kernel census attributes ~89
of ~108 runtime copy launches per generated token to this sequence, so this removes ~45 launches per token per
GPU (decode is launch-gap bound, QFP13). The copy is exact: outputs must be bit-identical.

## Hardware result (2026-10-04 review)

Profile v3 adoption ABBA (with 1307/1309/1310): +3% ~10K, +3.6% ~80K, greedy identical; kernels/token 1270 -> 1212 on its own.
