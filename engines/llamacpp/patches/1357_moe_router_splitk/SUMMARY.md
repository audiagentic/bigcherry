# 1357_moe_router_splitk

**Status:** validated
**Plan item:** QFP36

## What it does

The MoE router matmul (F32 weight `[n_embd, n_expert]` against a chunk of tokens, once per layer) runs through a
BigCherry split-K GEMM instead of rocBLAS SGEMM: K is cut into 8 parts, one workgroup per 64 x 64 output tile and
part, and a second kernel adds the parts in ascending order. Off by default; `BIGCHERRY_MOE_ROUTER_SPLITK=1` turns it
on.

## Why

In the kernel census of the released build (run `census-rel1-d24576`, Flash-Next, 75 chunks of 512 tokens) the router
SGEMM is 0.165 ms per call on an RX 7900 XTX and 0.432 ms on the R9700: 7.9 ms and 20.7 ms per chunk, about 1.9% of
prompt wall time on an XTX. For a 512 x 512 result SGEMM launches 64 workgroups that each walk all 2560 of K.

## Scope

Only the node `build_moe_ffn` marks as the router (op-param slot 6), F32 weight, activation and result, contiguous, at
least 64 experts, 64 tokens and 512 of K, AMD HIP. Other F32 matmuls are untouched. The sums are formed in another
order than SGEMM forms them: scores are equal within float tolerance, not bit-identical. Expert selection is
discontinuous near ties, so promotion needs identical output on hardware, not a score tolerance.

## Activation

`BIGCHERRY_PATCH_TRACE=1` prints `BIGCHERRY_PATCH_HIT patch=1357_moe_router_splitk mechanism=router-splitk k=
experts= tokens= parts=` once.

## Evidence

None yet. Offline mechanics tests only.
