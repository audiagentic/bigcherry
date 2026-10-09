# 1357_moe_router_splitk

Not promoted. Mechanism is in SUMMARY.md: the marked MoE router matmul runs through a split-K F32 GEMM with a
fixed-order reduction instead of rocBLAS SGEMM. Off by default; `BIGCHERRY_MOE_ROUTER_SPLITK=1` turns it on.

## Evidence

- Gate (kernel census of the released build, run `census-rel1-d24576`, Flash-Next, 75 chunks of 512 tokens): the
  router SGEMM runs once per layer per chunk on each target card, 0.165 ms per call on an RX 7900 XTX
  (`Cijk_..._MT64x64x8`) and 0.432 ms on the R9700 (`Cijk_..._MT16x16x16`). On the R9700, 3300 of its 3600 calls are
  followed directly by `topk_moe_cuda`, the rest by the unfused top-k kernels.
- Weight: `ffn_gate_inp.weight` is F32 `[2560, 512]` in all 48 layers of the UD-IQ4_XS model.

## Still to show

Offline mechanics and patch-lint, then on hardware: the activation marker, kernel time of the two new kernels against
SGEMM, an ABBA on one binary with `tools/lab/flash-next/queue-env-ab.sh` (flag on / off), and identical greedy output
at 8K, 24K and 98K. A changed output rejects the patch: expert selection must not move.
